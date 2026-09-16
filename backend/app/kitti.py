"""
Real sensor data: the KITTI raw dataset.

This module replaces every assumption the rest of the pipeline has to make with
a measurement, because KITTI ships the whole rig:

    image_02/        rectified colour camera, 1242 x 375
    velodyne_points/ HDL-64E LiDAR, ~120k points per frame
    oxts/            OXTS RT3003 GPS/IMU -- lat, lon, alt, roll, pitch, yaw
    calib_*.txt      the transforms that tie the three together

What that buys us
-----------------
The flat-ground projection in `geo.py` exists because a single camera cannot see
depth. Here we *do* have depth: project the LiDAR cloud into the image, take the
points that land inside a detection's box, and read the range straight off the
sensor. No ground-plane assumption, no guessed camera height. Objects with no
LiDAR return (too far, occluded, or glass) still fall back to the ground-plane
estimate, and are labelled differently so the two are never confused.

The transform chain
-------------------
Every frame moves a point through four frames of reference:

    LiDAR  --Tr_velo_to_cam-->  camera 0  --R_rect_00-->  rectified camera
                                                    --P_rect_02-->  image pixels

and, for world position, back out the other way:

    LiDAR  --Tr_imu_to_velo^-1-->  IMU  --R(roll,pitch,yaw)-->  ENU metres
                                                          --> WGS84 lat/lon

Conventions that are easy to get wrong, and are handled explicitly below:
  * KITTI's OXTS yaw is 0 = **east**, counter-clockwise positive. A compass
    heading is 0 = north, clockwise positive. They are not the same number.
  * The IMU body frame is x = forward, y = left, z = up.
  * `P_rect_02` has a non-zero fourth column: camera 2 is offset from the
    rectified origin by a baseline, which must not be dropped.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger("cadsoftware.kitti")

# A detection needs at least this many LiDAR returns inside its box before we
# trust the range. One or two stray points are as likely to be a reflection off
# something behind the object as the object itself.
MIN_LIDAR_POINTS = 8

# When separating an object from the background inside its bounding box, keep
# points within this distance of the nearest cluster.
FOREGROUND_DEPTH_MARGIN_M = 2.5


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def _read_calib_file(path: Path) -> dict[str, np.ndarray]:
    """Parse a KITTI calibration file into name -> float array."""
    out: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, values = line.partition(":")
        try:
            out[key.strip()] = np.array([float(v) for v in values.split()])
        except ValueError:
            continue  # calib_time and other non-numeric rows
    return out


def _rt_to_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 R and a 3x1 T."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.reshape(3, 3)
    matrix[:3, 3] = translation.reshape(3)
    return matrix


@dataclass
class Calibration:
    """The rig geometry, read from the three calib_*.txt files."""

    P_rect_02: np.ndarray       # 3x4, rectified camera 2 projection
    R_rect_00: np.ndarray       # 4x4, rectification rotation (homogeneous)
    Tr_velo_to_cam: np.ndarray  # 4x4, LiDAR -> camera 0
    Tr_imu_to_velo: np.ndarray  # 4x4, IMU -> LiDAR
    image_size: tuple[int, int]

    @classmethod
    def load(cls, calib_dir: Path) -> "Calibration":
        cam = _read_calib_file(calib_dir / "calib_cam_to_cam.txt")
        velo = _read_calib_file(calib_dir / "calib_velo_to_cam.txt")
        imu = _read_calib_file(calib_dir / "calib_imu_to_velo.txt")

        r_rect = np.eye(4)
        r_rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)

        size = cam.get("S_rect_02", cam.get("S_rect_00", np.array([1242.0, 375.0])))

        return cls(
            P_rect_02=cam["P_rect_02"].reshape(3, 4),
            R_rect_00=r_rect,
            Tr_velo_to_cam=_rt_to_matrix(velo["R"], velo["T"]),
            Tr_imu_to_velo=_rt_to_matrix(imu["R"], imu["T"]),
            image_size=(int(size[0]), int(size[1])),
        )

    # ------------------------------------------------------------ intrinsics

    @property
    def fx(self) -> float:
        return float(self.P_rect_02[0, 0])

    @property
    def fy(self) -> float:
        return float(self.P_rect_02[1, 1])

    @property
    def cx(self) -> float:
        return float(self.P_rect_02[0, 2])

    @property
    def cy(self) -> float:
        return float(self.P_rect_02[1, 2])

    @property
    def hfov_deg(self) -> float:
        width = self.image_size[0]
        return math.degrees(2.0 * math.atan(width / (2.0 * self.fx)))

    # ------------------------------------------------------------ projection

    @property
    def velo_to_image(self) -> np.ndarray:
        """3x4 matrix taking a homogeneous LiDAR point straight to image pixels."""
        return self.P_rect_02 @ self.R_rect_00 @ self.Tr_velo_to_cam

    def project_lidar(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project LiDAR points into the image.

        Returns (uv, depth, keep) where `keep` masks the points that are in
        front of the camera and land inside the frame. Points behind the camera
        would otherwise project to perfectly plausible-looking pixels, which is
        one of the classic ways to get a projection subtly wrong.
        """
        xyz = points[:, :3]
        homogeneous = np.hstack([xyz, np.ones((len(xyz), 1))])
        projected = homogeneous @ self.velo_to_image.T  # (N, 3)

        depth = projected[:, 2]
        in_front = depth > 0.5  # metres; also avoids dividing by ~0

        uv = np.zeros((len(xyz), 2))
        safe = np.where(in_front, depth, 1.0)
        uv[:, 0] = projected[:, 0] / safe
        uv[:, 1] = projected[:, 1] / safe

        width, height = self.image_size
        inside = (
            in_front
            & (uv[:, 0] >= 0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        )
        return uv, depth, inside

    # ------------------------------------------------------------ rig geometry

    @property
    def camera_origin_in_velo(self) -> np.ndarray:
        """Where the camera sits in LiDAR coordinates.

        Used to work out the camera's height above the road, which the
        ground-plane fallback needs and which we would otherwise have to guess.
        """
        inverse = np.linalg.inv(self.Tr_velo_to_cam)
        return inverse[:3, 3]

    def velo_to_imu(self, points_velo: np.ndarray) -> np.ndarray:
        """LiDAR coordinates to the IMU body frame (x forward, y left, z up)."""
        inverse = np.linalg.inv(self.Tr_imu_to_velo)
        homogeneous = np.hstack([points_velo, np.ones((len(points_velo), 1))])
        return (homogeneous @ inverse.T)[:, :3]

    def camera_axis_in_imu(self) -> np.ndarray:
        """The optical axis as a unit vector in the IMU frame.

        The camera does not point exactly along the vehicle's nose, so the
        ground-plane fallback derives its pitch and heading from this rather
        than assuming the two are aligned.
        """
        # +Z in the rectified camera frame, taken back to LiDAR then to IMU.
        axis_cam = np.array([0.0, 0.0, 1.0])
        rot_velo_to_rect = (self.R_rect_00 @ self.Tr_velo_to_cam)[:3, :3]
        axis_velo = np.linalg.inv(rot_velo_to_rect) @ axis_cam
        axis_imu = np.linalg.inv(self.Tr_imu_to_velo)[:3, :3] @ axis_velo
        return axis_imu / np.linalg.norm(axis_imu)


# --------------------------------------------------------------------------
# OXTS GPS / IMU
# --------------------------------------------------------------------------


@dataclass
class OxtsPose:
    """One GPS/IMU packet: where the vehicle was and how it was oriented."""

    lat: float
    lon: float
    alt: float
    roll: float    # radians
    pitch: float   # radians
    yaw: float     # radians, 0 = EAST, counter-clockwise positive
    vf: float      # forward velocity, m/s
    vl: float
    vu: float

    @classmethod
    def parse(cls, path: Path) -> "OxtsPose":
        values = [float(v) for v in path.read_text().split()]
        return cls(
            lat=values[0], lon=values[1], alt=values[2],
            roll=values[3], pitch=values[4], yaw=values[5],
            vf=values[8], vl=values[9], vu=values[10],
        )

    @property
    def heading_deg(self) -> float:
        """Compass heading: 0 = north, clockwise positive.

        OXTS measures yaw from east, counter-clockwise. Converting is a
        reflection plus a quarter turn, not just an offset -- getting this
        backwards mirrors the whole scene about the north-south axis.
        """
        return (90.0 - math.degrees(self.yaw)) % 360.0

    @property
    def speed_kmh(self) -> float:
        return math.hypot(self.vf, self.vl) * 3.6

    def rotation_to_enu(self) -> np.ndarray:
        """Rotate a vector from the IMU body frame into world ENU.

        Standard aerospace composition Rz(yaw) @ Ry(pitch) @ Rx(roll), which
        works out neatly here because OXTS yaw is already measured from east and
        ENU's X axis is east: at yaw = 0 the vehicle's nose maps to +X.
        """
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)

        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return rz @ ry @ rx


# --------------------------------------------------------------------------
# Sequence
# --------------------------------------------------------------------------


@dataclass
class KittiFrame:
    index: int
    image_path: Path
    velodyne_path: Path
    oxts_path: Path

    def pose(self) -> OxtsPose:
        return OxtsPose.parse(self.oxts_path)

    def lidar(self) -> np.ndarray:
        """The raw point cloud as an (N, 4) array of x, y, z, reflectance."""
        return np.fromfile(self.velodyne_path, dtype=np.float32).reshape(-1, 4)


@dataclass
class KittiSequence:
    """One KITTI raw drive, with its calibration."""

    name: str
    root: Path            # the drive directory, e.g. 2011_09_26_drive_0048_sync
    calib: Calibration
    frames: list[KittiFrame]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        return 10.0  # KITTI raw is captured at 10 Hz

    def summary(self) -> dict:
        first = self.frames[0].pose()
        return {
            "name": self.name,
            "frames": self.frame_count,
            "fps": self.fps,
            "duration_s": round(self.frame_count / self.fps, 2),
            "image_size": list(self.calib.image_size),
            "origin": {"lat": first.lat, "lon": first.lon, "alt": first.alt},
            "hfov_deg": round(self.calib.hfov_deg, 2),
            "focal_px": round(self.calib.fx, 1),
            "principal_point": [round(self.calib.cx, 1), round(self.calib.cy, 1)],
        }


def discover(data_dir: Path) -> list[KittiSequence]:
    """Find every KITTI drive under data/kitti.

    A drive is any *_sync directory that has all three sensors and a sibling
    calibration set. Anything incomplete is skipped rather than half-loaded.
    """
    sequences: list[KittiSequence] = []
    if not data_dir.exists():
        return sequences

    for date_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if not (date_dir / "calib_cam_to_cam.txt").exists():
            continue
        try:
            calib = Calibration.load(date_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: unreadable calibration (%s)", date_dir.name, exc)
            continue

        for drive in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            images = sorted((drive / "image_02" / "data").glob("*.png"))
            velo = drive / "velodyne_points" / "data"
            oxts = drive / "oxts" / "data"
            if not images or not velo.is_dir() or not oxts.is_dir():
                continue

            frames: list[KittiFrame] = []
            for i, image_path in enumerate(images):
                stem = image_path.stem
                velo_path = velo / f"{stem}.bin"
                oxts_path = oxts / f"{stem}.txt"
                if velo_path.exists() and oxts_path.exists():
                    frames.append(KittiFrame(i, image_path, velo_path, oxts_path))

            if frames:
                sequences.append(KittiSequence(drive.name, drive, calib, frames))

    return sequences


@lru_cache(maxsize=4)
def _cached_discover(data_dir_str: str, stamp: float) -> tuple[KittiSequence, ...]:
    return tuple(discover(Path(data_dir_str)))


def find_sequence(data_dir: Path, name: str) -> Optional[KittiSequence]:
    for sequence in discover(data_dir):
        if sequence.name == name:
            return sequence
    return None


# --------------------------------------------------------------------------
# Depth from LiDAR
# --------------------------------------------------------------------------


@dataclass
class LidarFix:
    """A measured position for one detection, or a note about why there isn't one."""

    ok: bool
    east_m: float = 0.0
    north_m: float = 0.0
    up_m: float = 0.0
    depth_m: float = 0.0        # range along the camera's optical axis
    distance_m: float = 0.0     # horizontal ground distance from the vehicle
    point_count: int = 0
    width_m: float = 0.0
    height_m: float = 0.0
    reason: str = ""


def ground_level_velo(points: np.ndarray) -> Optional[float]:
    """Height of the road surface in LiDAR coordinates, in metres.

    Rather than assuming the documented sensor height, measure it: take the
    returns from a patch of road ahead of the vehicle and use the median z.
    The median shrugs off the cars and kerbs that inevitably fall inside the
    patch, which a mean would not.
    """
    ahead = points[
        (points[:, 0] > 5.0) & (points[:, 0] < 30.0) & (np.abs(points[:, 1]) < 4.0)
    ]
    if len(ahead) < 200:
        return None
    return float(np.median(ahead[:, 2]))


def measure_detections(
    boxes: list[tuple[float, float, float, float]],
    points: np.ndarray,
    calib: Calibration,
    pose: OxtsPose,
) -> list[LidarFix]:
    """Measure each detection's real position from the LiDAR cloud.

    For every bounding box:
      1. keep the LiDAR points that project inside it
      2. separate the object from whatever is behind it
      3. take the median of the foreground points as its position
      4. rotate that into world ENU using the GPS/IMU orientation

    Step 2 is the one that matters. A box drawn around a pedestrian also
    contains returns from the wall twenty metres behind them, and a naive mean
    would place the pedestrian somewhere in between. Taking a low percentile of
    the depths and keeping only what is close to it isolates the nearest
    surface, which is the object we actually detected.
    """
    uv, depth, inside = calib.project_lidar(points)

    valid_uv = uv[inside]
    valid_depth = depth[inside]
    valid_xyz = points[inside][:, :3]

    rotation = pose.rotation_to_enu()
    fixes: list[LidarFix] = []

    for x1, y1, x2, y2 in boxes:
        in_box = (
            (valid_uv[:, 0] >= x1) & (valid_uv[:, 0] <= x2)
            & (valid_uv[:, 1] >= y1) & (valid_uv[:, 1] <= y2)
        )
        count = int(in_box.sum())
        if count < MIN_LIDAR_POINTS:
            fixes.append(LidarFix(
                ok=False, point_count=count,
                reason=f"only {count} LiDAR returns inside the box",
            ))
            continue

        box_depth = valid_depth[in_box]
        box_xyz = valid_xyz[in_box]

        # --- separate foreground from background -------------------------
        near = float(np.percentile(box_depth, 20))
        foreground = box_depth <= near + FOREGROUND_DEPTH_MARGIN_M
        if foreground.sum() < MIN_LIDAR_POINTS // 2:
            foreground = box_depth <= near + FOREGROUND_DEPTH_MARGIN_M * 2

        object_points = box_xyz[foreground]
        centre_velo = np.median(object_points, axis=0)

        # --- real extents, straight off the point cloud -------------------
        spread = object_points.max(axis=0) - object_points.min(axis=0)
        width_m = float(math.hypot(spread[0], spread[1]))
        height_m = float(spread[2])

        # --- LiDAR frame -> IMU body frame -> world ENU --------------------
        centre_imu = calib.velo_to_imu(centre_velo.reshape(1, 3))[0]
        enu = rotation @ centre_imu

        fixes.append(LidarFix(
            ok=True,
            east_m=float(enu[0]), north_m=float(enu[1]), up_m=float(enu[2]),
            depth_m=float(np.median(box_depth[foreground])),
            distance_m=float(math.hypot(enu[0], enu[1])),
            point_count=int(foreground.sum()),
            width_m=width_m,
            height_m=height_m,
        ))

    return fixes


@dataclass
class KittiFrameResult:
    """One processed frame: detections, their measured positions, and the pose.

    The camera pose is per frame because the vehicle is moving -- its GPS
    position and heading change every 100 ms, which is the whole point of using
    a real drive rather than a static shot.
    """

    index: int
    time_s: float
    detections: list           # app.detector.Detection, pixel space
    fixes: list[LidarFix]      # parallel to detections
    camera: dict               # pose derived from calibration + OXTS
    lidar_points: int
    ground_z: Optional[float]
    speed_kmh: float


@dataclass
class KittiResult:
    """Everything a processed sequence produced.

    Field names deliberately mirror `video.VideoResult` so the same job-status
    endpoint can report on either without special-casing.
    """

    width: int
    height: int
    source_fps: float
    duration_s: float
    frames: list[KittiFrameResult]
    processing_ms: float
    target_fps: float
    sequence: str
    calibration: dict

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


def process_sequence(
    sequence: KittiSequence,
    detector,
    out_dir: Path,
    confidence: float = 0.35,
    iou: float = 0.45,
    job=None,
) -> KittiResult:
    """Run the full real-sensor pipeline over a KITTI drive.

    Per frame: detect with YOLO (tracked, so identities persist), load the
    Velodyne cloud, measure each detection's true position from it, and derive
    the camera pose from the calibration plus the GPS/IMU packet.

    Nothing here is assumed that the sensors can tell us instead.
    """
    import time

    from .video import encode_frames_to_video

    started = time.perf_counter()

    if job is not None:
        job.message = "loading tracker"
    model = detector.new_tracking_model()

    frames: list[KittiFrameResult] = []
    total = sequence.frame_count

    for i, frame in enumerate(sequence.frames):
        if job is not None:
            job.progress = min(0.92, i / max(1, total))
            job.message = f"fusing frame {i + 1}/{total}"

        image = np.asarray(Image.open(frame.image_path).convert("RGB"))

        # persist=True carries track identities from one frame to the next.
        results = model.track(
            source=image, persist=True, tracker="bytetrack.yaml",
            conf=confidence, iou=iou, verbose=False,
        )
        detections = _to_detections(results, image.shape[1], image.shape[0])

        points = frame.lidar()
        pose = frame.pose()
        ground_z = ground_level_velo(points)

        boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        fixes = measure_detections(boxes, points, sequence.calib, pose)

        frames.append(KittiFrameResult(
            index=frame.index,
            time_s=frame.index / sequence.fps,
            detections=detections,
            fixes=fixes,
            camera=camera_pose_for_fallback(sequence.calib, pose, ground_z),
            lidar_points=len(points),
            ground_z=ground_z,
            speed_kmh=pose.speed_kmh,
        ))

    # Encode the frames so the existing video transport can play them back.
    if job is not None:
        job.message = "encoding playback clip"
    video_path, codec = encode_frames_to_video(
        [f.image_path for f in sequence.frames], out_dir, sequence.name, sequence.fps,
    )
    if job is not None:
        job.path = str(video_path)
        job.url = f"/uploads/{video_path.name}"

    calib = sequence.calib
    return KittiResult(
        width=calib.image_size[0],
        height=calib.image_size[1],
        source_fps=sequence.fps,
        duration_s=total / sequence.fps,
        frames=frames,
        processing_ms=(time.perf_counter() - started) * 1000.0,
        target_fps=sequence.fps,
        sequence=sequence.name,
        calibration={
            "fx": round(calib.fx, 3), "fy": round(calib.fy, 3),
            "cx": round(calib.cx, 3), "cy": round(calib.cy, 3),
            "hfov_deg": round(calib.hfov_deg, 2),
            "image_size": list(calib.image_size),
            "codec": codec,
        },
    )


def _to_detections(results, width: int, height: int) -> list:
    """Ultralytics tracking output to our Detection objects."""
    from .detector import Detection

    if not results:
        return []
    boxes = results[0].boxes
    names = results[0].names
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    out = []
    for new_id, i in enumerate(np.argsort(-confs)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        out.append(Detection(
            id=new_id,
            class_id=int(clss[i]),
            class_name=str(names[int(clss[i])]),
            confidence=float(confs[i]),
            x1=max(0.0, x1), y1=max(0.0, y1),
            x2=min(float(width), x2), y2=min(float(height), y2),
            track_id=int(ids[i]) if ids is not None else None,
        ))
    return out


def camera_pose_for_fallback(
    calib: Calibration, pose: OxtsPose, ground_z: Optional[float],
) -> dict:
    """Camera pose in the form the ground-plane fallback in `geo.py` expects.

    Everything here is derived from the rig and the GPS/IMU rather than typed
    in: the height comes from the measured road surface, and the pitch and
    heading from the optical axis rotated into world coordinates.
    """
    # Height of the camera above the road, measured rather than assumed.
    camera_velo = calib.camera_origin_in_velo
    if ground_z is not None:
        altitude = float(camera_velo[2] - ground_z)
    else:
        # No usable road patch in this frame: fall back to the rig's nominal
        # LiDAR height, which KITTI documents as 1.73 m above ground.
        altitude = float(camera_velo[2] + 1.73)

    axis_enu = pose.rotation_to_enu() @ calib.camera_axis_in_imu()
    heading = (math.degrees(math.atan2(axis_enu[0], axis_enu[1]))) % 360.0
    pitch_below = math.degrees(math.asin(max(-1.0, min(1.0, -axis_enu[2]))))

    return {
        "altitude_m": max(0.2, altitude),
        "heading_deg": heading,
        "pitch_deg": max(0.0, pitch_below),
        "hfov_deg": calib.hfov_deg,
        "fx_px": calib.fx,
        "fy_px": calib.fy,
        "cx_px": calib.cx,
        "cy_px": calib.cy,
        "lat": pose.lat,
        "lon": pose.lon,
    }
