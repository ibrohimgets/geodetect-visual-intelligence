"""
Video mode: tracking objects across frames.

    video frame -> YOLO -> tracker -> coordinates -> map / 3D update

A still image gives you a set of boxes. A video gives you the same boxes plus
*identity*: the tracker recognises that the car in frame 40 is the car from
frame 1, so it keeps the same id. That identity is what makes motion trails and
per-object history possible, and it is the only genuinely new idea in this
module -- everything downstream is the same geometry the image path uses.

Design notes
------------
Processing runs as a background job with progress, because a minute of video is
hundreds of inferences and an HTTP request should not sit open for that long.

Only *pixel-space* results are stored. Geo-referencing happens afterwards, on
demand, exactly as it does for a still image -- so moving the camera sliders
re-projects a whole tracked video without re-running the detector.

Frames are sampled down to a few per second. Tracking at the source frame rate
would be slower for no benefit: objects barely move between consecutive frames
of 30 fps footage, and the trails look the same.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .detector import Detection, Detector

log = logging.getLogger("cadsoftware.video")

# Ceilings, so an hour-long upload cannot wedge the machine.
MAX_PROCESSED_FRAMES = 300
MAX_DURATION_S = 180.0
DEFAULT_TARGET_FPS = 5.0


@dataclass
class VideoFrame:
    index: int          # index in the source video
    time_s: float       # playback position, for syncing with the <video> element
    detections: list[Detection]


@dataclass
class VideoResult:
    width: int
    height: int
    source_fps: float
    duration_s: float
    frames: list[VideoFrame]
    processing_ms: float
    target_fps: float

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def track_ids(self) -> list[int]:
        seen: list[int] = []
        for frame in self.frames:
            for det in frame.detections:
                if det.track_id is not None and det.track_id not in seen:
                    seen.append(det.track_id)
        return seen


@dataclass
class VideoJob:
    id: str
    filename: str
    path: str
    url: str
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0           # 0..1
    message: str = ""
    result: Optional[VideoResult] = None
    error: str = ""
    created: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        """The polling payload. Deliberately small -- the timeline is fetched
        separately once the job is done."""
        out: dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "filename": self.filename,
            "url": self.url,
        }
        if self.status == "error":
            out["error"] = self.error
        if self.status == "done" and self.result:
            out.update({
                "width": self.result.width,
                "height": self.result.height,
                "duration_s": round(self.result.duration_s, 2),
                "source_fps": round(self.result.source_fps, 2),
                "processed_frames": self.result.frame_count,
                "processing_ms": round(self.result.processing_ms, 1),
                "track_count": len(self.result.track_ids()),
            })
        return out


class VideoProcessor:
    """Owns the job registry and does the frame-by-frame work."""

    def __init__(self, detector: Detector, max_jobs: int = 8):
        self.detector = detector
        self._jobs: dict[str, VideoJob] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    # ------------------------------------------------------------------ jobs

    def get(self, job_id: str) -> Optional[VideoJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(
        self,
        path: Path,
        filename: str,
        url: str,
        confidence: float = 0.35,
        iou: float = 0.45,
        target_fps: float = DEFAULT_TARGET_FPS,
    ) -> VideoJob:
        job = VideoJob(
            id=uuid.uuid4().hex[:16],
            filename=filename,
            path=str(path),
            url=url,
        )

        with self._lock:
            self._jobs[job.id] = job
            # Drop the oldest finished jobs, and their files with them.
            if len(self._jobs) > self._max_jobs:
                done = sorted(
                    (j for j in self._jobs.values() if j.status in ("done", "error")),
                    key=lambda j: j.created,
                )
                for old in done[: len(self._jobs) - self._max_jobs]:
                    self._jobs.pop(old.id, None)
                    try:
                        Path(old.path).unlink(missing_ok=True)
                    except OSError:
                        pass

        thread = threading.Thread(
            target=self._run,
            args=(job, confidence, iou, target_fps),
            daemon=True,
            name=f"video-{job.id}",
        )
        thread.start()
        return job

    def submit_worker(self, worker, filename: str, url: str, path: str = "") -> VideoJob:
        """Run an arbitrary worker under the same job, progress and polling machinery.

        The KITTI pipeline is a different kind of work -- a folder of frames with
        their own sensors, not a video file to decode -- but it wants exactly the
        same treatment: background thread, progress the UI can poll, results kept
        for later. Rather than duplicate all of that, it hands in a callable.

        The worker receives the job so it can report progress, and returns
        whatever result object it likes.
        """
        job = VideoJob(id=uuid.uuid4().hex[:16], filename=filename, path=path, url=url)
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            job.status = "running"
            try:
                job.result = worker(job)
                job.progress = 1.0
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - reported through the job
                log.exception("Worker job %s failed", job.id)
                job.status = "error"
                job.error = str(exc)
                job.message = "failed"

        threading.Thread(target=run, daemon=True, name=f"job-{job.id}").start()
        return job

    # ------------------------------------------------------------- processing

    def _run(self, job: VideoJob, confidence: float, iou: float, target_fps: float) -> None:
        job.status = "running"
        job.message = "opening video"
        started = time.perf_counter()

        capture = None
        try:
            capture = cv2.VideoCapture(job.path)
            if not capture.isOpened():
                raise RuntimeError("could not open the video file (unsupported codec?)")

            source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            if not (1.0 <= source_fps <= 240.0):
                source_fps = 30.0  # some containers report nonsense
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = total_frames / source_fps if total_frames > 0 else 0.0

            if duration > MAX_DURATION_S:
                log.info("Video is %.0fs; only the first %.0fs will be processed",
                         duration, MAX_DURATION_S)

            # Sample down to roughly target_fps, then cap the total.
            step = max(1, int(round(source_fps / max(0.5, target_fps))))
            budget_frames = min(
                MAX_PROCESSED_FRAMES,
                int(MAX_DURATION_S * source_fps / step) or MAX_PROCESSED_FRAMES,
            )

            # A private model instance, so this job's track ids are its own.
            job.message = "loading tracker"
            model = self.detector.new_tracking_model()

            frames: list[VideoFrame] = []
            index = 0
            processed = 0

            while processed < budget_frames:
                ok = capture.grab()          # cheap: decode only what we keep
                if not ok:
                    break

                if index % step == 0:
                    ok, frame_bgr = capture.retrieve()
                    if not ok:
                        break

                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    detections = self._track_frame(model, rgb, confidence, iou)
                    frames.append(
                        VideoFrame(
                            index=index,
                            time_s=index / source_fps,
                            detections=detections,
                        )
                    )
                    processed += 1

                    if total_frames > 0:
                        job.progress = min(0.99, index / min(total_frames,
                                                             budget_frames * step))
                    else:
                        job.progress = min(0.99, processed / budget_frames)
                    job.message = f"tracking frame {processed}/{budget_frames}"

                index += 1

            if not frames:
                raise RuntimeError("no frames could be decoded from this video")

            job.result = VideoResult(
                width=width,
                height=height,
                source_fps=source_fps,
                duration_s=duration,
                frames=frames,
                processing_ms=(time.perf_counter() - started) * 1000.0,
                target_fps=source_fps / step,
            )
            job.progress = 1.0
            job.status = "done"
            job.message = f"{len(frames)} frames, {len(job.result.track_ids())} tracks"
            log.info("Video %s done: %s", job.id, job.message)

        except Exception as exc:  # noqa: BLE001 - reported through the job
            log.exception("Video job %s failed", job.id)
            job.status = "error"
            job.error = str(exc)
            job.message = "failed"
        finally:
            if capture is not None:
                capture.release()

    def _track_frame(self, model, rgb: np.ndarray, confidence: float,
                     iou: float) -> list[Detection]:
        """One frame through the tracker.

        `persist=True` is what carries identities forward: it tells Ultralytics
        this frame continues the previous one rather than starting a new clip.
        """
        results = model.track(
            source=rgb,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            iou=iou,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        names = results[0].names
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        # The tracker only assigns ids once it is confident; early frames and
        # low-quality detections can come back without one.
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        h, w = rgb.shape[:2]
        out: list[Detection] = []
        order = np.argsort(-confs)
        for new_id, i in enumerate(order):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            out.append(
                Detection(
                    id=new_id,
                    class_id=int(clss[i]),
                    class_name=str(names[int(clss[i])]),
                    confidence=float(confs[i]),
                    x1=max(0.0, x1), y1=max(0.0, y1),
                    x2=min(float(w), x2), y2=min(float(h), y2),
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return out


def encode_frames_to_video(
    image_paths: list[Path],
    out_dir: Path,
    stem: str,
    fps: float = 10.0,
) -> tuple[Path, str]:
    """Encode a folder of still frames into a clip the browser can play.

    A dataset like KITTI ships numbered PNGs, not a video. Rather than build a
    second playback path in the frontend for image sequences, we encode them
    once and reuse the existing video transport untouched.

    Codec choice is not cosmetic: OpenCV will cheerfully write MPEG-4 Part 2,
    which no browser decodes, and report success while doing it. So each
    candidate is written and then the FOURCC is read back out of the finished
    file, and only a tag the browser can actually play is accepted.
    """
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"could not read {image_paths[0]}")
    height, width = first.shape[:2]

    candidates = [("avc1", ".mp4"), ("VP80", ".webm"), ("VP90", ".webm")]
    browser_safe = {"h264", "avc1", "vp80", "vp90", "vp08", "vp09"}

    for fourcc, ext in candidates:
        out_path = out_dir / f"{stem}{ext}"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
        )
        if not writer.isOpened():
            writer.release()
            continue

        for path in image_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        writer.release()

        # Verify what actually landed in the container.
        check = cv2.VideoCapture(str(out_path))
        raw = int(check.get(cv2.CAP_PROP_FOURCC)) if check.isOpened() else 0
        check.release()
        tag = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)).strip().lower()

        if tag in browser_safe:
            log.info("Encoded %d frames to %s (codec %s)", len(image_paths), out_path.name, tag)
            return out_path, tag
        out_path.unlink(missing_ok=True)

    raise RuntimeError("no browser-playable codec available from OpenCV")


def build_trails(
    frames: list[dict],
    max_points: int = 60,
) -> list[dict[str, Any]]:
    """Turn per-frame geo-referenced detections into one path per tracked object.

    `frames` are the already geo-referenced frames, so a trail is simply every
    position a given track id was seen at, in time order. Trails are what make
    movement legible on the map and in the 3D scene.
    """
    paths: dict[int, dict[str, Any]] = {}

    for frame in frames:
        for det in frame["detections"]:
            track_id = det.get("track_id")
            world = det.get("world", {})
            if track_id is None or not world.get("valid"):
                continue

            path = paths.setdefault(track_id, {
                "track_id": track_id,
                "class_name": det["class_name"],
                "class_id": det["class_id"],
                "label": f"{det['class_name'].replace(' ', '_')}_{track_id:02d}",
                "points": [],
            })
            path["points"].append({
                "t": round(frame["time_s"], 3),
                "lat": world["lat"],
                "lon": world["lon"],
                "east_m": world["east_m"],
                "north_m": world["north_m"],
            })

    trails: list[dict[str, Any]] = []
    for path in paths.values():
        points = path["points"]
        # Thin very long trails so the map stays responsive, keeping the ends.
        if len(points) > max_points:
            stride = len(points) / max_points
            points = [points[int(i * stride)] for i in range(max_points - 1)]
            points.append(path["points"][-1])
            path["points"] = points

        path["distance_m"] = _path_length(path["points"])
        path["displacement_m"] = _displacement(path["points"])
        path["first_seen_s"] = path["points"][0]["t"] if path["points"] else 0.0
        path["last_seen_s"] = path["points"][-1]["t"] if path["points"] else 0.0
        trails.append(path)

    trails.sort(key=lambda p: -p["displacement_m"])
    return trails


def _metres_between(a: dict, b: dict) -> float:
    """Ground distance between two trail points, from their WGS84 coordinates."""
    lat_mid = math.radians((a["lat"] + b["lat"]) / 2.0)
    north = (b["lat"] - a["lat"]) * 111132.0
    east = (b["lon"] - a["lon"]) * 111320.0 * math.cos(lat_mid)
    return math.hypot(east, north)


def _path_length(points: list[dict]) -> float:
    """Total ground distance travelled along a trail, in metres.

    Measured from latitude and longitude, deliberately **not** from the ENU
    columns. ENU is relative to the camera, and on a moving platform -- a KITTI
    drive, say -- a parked car's ENU position sweeps right past the vehicle
    while the car itself never moves. Using ENU here would report the ego's
    motion as the object's, which is exactly backwards.
    """
    return round(sum(_metres_between(a, b) for a, b in zip(points, points[1:])), 2)


def _displacement(points: list[dict]) -> float:
    """Straight-line distance from where a track started to where it ended.

    Reported alongside the path length because together they separate a genuine
    journey from a stationary object jittering under GPS and detection noise: a
    parked car accumulates path length but almost no displacement.
    """
    if len(points) < 2:
        return 0.0
    return round(_metres_between(points[0], points[-1]), 2)
