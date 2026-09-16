"""
FastAPI application: the visual-intelligence pipeline, end to end.

    image / video -> detection (+tracking) -> camera geometry
                  -> estimated world coordinates -> GIS map -> 3D scene
                  -> scene graph -> language assistant

Route map
---------
    GET  /                        the single-page frontend
    GET  /api/health              model, device and assistant status
    GET  /api/classes             the 80 COCO classes the detector knows
    GET  /api/categories          class groupings used by the filter chips
    GET  /api/samples             bundled demo media

    POST /api/analyze             upload an image: detect + geo-reference
    POST /api/reproject           redo only the geometry with a new camera pose
    POST /api/scene               scene graph for an arbitrary set of detections

    POST /api/video/analyze       upload a video: starts a tracking job
    GET  /api/video/job/{id}      poll job progress
    POST /api/video/timeline      geo-reference a finished job's tracked frames

    POST /api/assistant           ask a question about the current scene
    POST /api/export/{fmt}        geojson | json | csv

    GET  /docs                    interactive API documentation

Why analyze and reproject are separate
--------------------------------------
Inference is slow and depends only on the pixels. Geo-referencing is arithmetic
and depends only on the camera pose. Splitting them lets the operator drag the
altitude or heading sliders and watch the map and 3D scene update immediately,
without paying for detection again -- and it mirrors the fact that the stages
really are independent. Video works the same way: tracking runs once, and the
whole timeline can be re-projected instantly.
"""

from __future__ import annotations

import io
import json
import math
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from . import analytics, config, exports, kitti, llm
from . import scene_graph as sg
from . import video as video_mod
from .categories import category_summary
from .detector import GROUND_CLASSES, Detection, DetectionResult, Detector
from .exif import extract_camera_hints
from .geo import CameraModel, ground_footprint, project_detection
from .schemas import (
    BBox,
    CameraOut,
    CameraParams,
    ClassesResponse,
    DetectionOut,
    DetectParams,
    DetectResponse,
    FootprintPoint,
    HealthResponse,
    ImageInfo,
    WorldPosition,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cadsoftware")

detector = Detector(weights=config.MODEL_WEIGHTS, models_dir=config.MODELS_DIR)
video_processor = video_mod.VideoProcessor(detector)


# --------------------------------------------------------------------------
# Detection cache
# --------------------------------------------------------------------------
# Keeps raw pixel-space detections for recently analysed images so
# /api/reproject can redo the geometry without touching the neural network.

_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_LOCK = Lock()
_CACHE_MAX = 32


def _cache_put(image_id: str, payload: dict) -> None:
    with _CACHE_LOCK:
        _CACHE[image_id] = payload
        _CACHE.move_to_end(image_id)
        while len(_CACHE) > _CACHE_MAX:
            old_id, old = _CACHE.popitem(last=False)
            try:
                Path(old["path"]).unlink(missing_ok=True)
            except OSError:
                log.debug("Could not remove expired upload for %s", old_id)


def _cache_get(image_id: str) -> dict:
    with _CACHE_LOCK:
        entry = _CACHE.get(image_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail="Unknown or expired image id. Upload and analyse the image again.",
            )
        _CACHE.move_to_end(image_id)
        return entry


# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up - loading detection model %s", config.MODEL_WEIGHTS)
    try:
        detector.load()
        log.info("Model loaded, %d classes available", len(detector.class_names))
    except Exception as exc:  # noqa: BLE001 - reported via /api/health
        log.error("Model did not load: %s", exc)

    state = llm.status()
    log.info("Assistant: %s (%s)",
             "ready" if state["available"] else "unavailable",
             state["model"] if state["available"] else state["reason"])
    yield
    log.info("Shutting down")


app = FastAPI(
    title="GeoDetect - Visual Intelligence Pipeline",
    description=(
        "Detect objects in an image or video, estimate their real-world positions "
        "from the camera geometry, and explore the result on a map, in 3D, and "
        "through a language assistant that reasons over the measurements."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _build_camera(params: CameraParams, width: int, height: int) -> CameraModel:
    return CameraModel(
        image_width=width,
        image_height=height,
        hfov_deg=params.hfov_deg,
        altitude_m=params.altitude_m,
        pitch_deg=params.pitch_deg,
        heading_deg=params.heading_deg,
        roll_deg=params.roll_deg,
        lat=params.lat,
        lon=params.lon,
        max_range_m=params.max_range_m,
    )


def _camera_out(cam: CameraModel) -> CameraOut:
    return CameraOut(
        lat=cam.lat, lon=cam.lon, altitude_m=cam.altitude_m,
        pitch_deg=cam.pitch_deg, heading_deg=cam.heading_deg, roll_deg=cam.roll_deg,
        hfov_deg=cam.hfov_deg, vfov_deg=cam.vfov_deg,
        fx=cam.fx, fy=cam.fy, max_range_m=cam.max_range_m,
    )


def _georeference(detections: list[Detection], cam: CameraModel) -> list[DetectionOut]:
    """The geometry stage: give every pixel-space box a world position."""
    out: list[DetectionOut] = []
    for det in detections:
        fix = project_detection(cam, det.x1, det.y1, det.x2, det.y2)
        cx, cy = det.center
        ax, ay = det.ground_anchor

        out.append(
            DetectionOut(
                id=det.id,
                track_id=det.track_id,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=round(det.confidence, 4),
                is_ground_class=det.is_ground_class,
                bbox=BBox(
                    x1=round(det.x1, 1), y1=round(det.y1, 1),
                    x2=round(det.x2, 1), y2=round(det.y2, 1),
                    width=round(det.width, 1), height=round(det.height, 1),
                    cx=round(cx, 1), cy=round(cy, 1),
                    anchor_x=round(ax, 1), anchor_y=round(ay, 1),
                ),
                world=WorldPosition(
                    valid=fix.valid,
                    reason=fix.reason,
                    east_m=round(fix.east_m, 3),
                    north_m=round(fix.north_m, 3),
                    up_m=0.0,
                    lat=fix.lat, lon=fix.lon,
                    ground_range_m=round(fix.ground_range_m, 2),
                    slant_range_m=round(fix.slant_range_m, 2),
                    bearing_deg=round(fix.bearing_deg, 1),
                    est_width_m=round(fix.width_m, 2),
                    est_height_m=round(fix.height_m, 2),
                ),
            )
        )
    return out


def _scene_payload(detections: list[dict], camera: dict) -> dict[str, Any]:
    """Scene graph in both the structured and the human-readable form."""
    graph = sg.build(detections, camera)
    return {"tree": graph.to_tree(), "context": graph.to_context()}


def _load_image(raw: bytes, filename: str) -> tuple[np.ndarray, Image.Image]:
    """Decode an upload into an RGB array, honouring EXIF rotation.

    Phones and drones often store a sideways sensor read plus an orientation
    tag. Ignoring it means detection runs on a rotated image and every
    coordinate downstream is wrong, so we bake the rotation in first. Oversized
    photos are scaled down, and because we serve this same processed image to
    the browser, pixel coordinates always match what the operator sees.
    """
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"Could not read '{filename}' as an image: {exc}"
        ) from exc

    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")

    longest = max(img.size)
    if longest > config.MAX_IMAGE_DIM:
        scale = config.MAX_IMAGE_DIM / longest
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )

    return np.asarray(img), img


# --------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    device = "cpu"
    num_classes = 0
    error = detector.load_error
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        pass

    if detector.is_loaded:
        try:
            num_classes = len(detector.class_names)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

    return HealthResponse(
        status="ok" if detector.is_loaded else "degraded",
        model_loaded=detector.is_loaded,
        model_name=config.MODEL_WEIGHTS,
        device=device,
        num_classes=num_classes,
        error=error,
        assistant=llm.status(),
    )


@app.get("/api/classes", response_model=ClassesResponse, tags=["system"])
def classes() -> ClassesResponse:
    try:
        names = detector.class_names
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    return ClassesResponse(classes=names, ground_classes=sorted(GROUND_CLASSES))


@app.get("/api/categories", tags=["system"])
def categories():
    """Class groupings for the filter chips: People, Vehicles, Animals, and so on."""
    return {"categories": category_summary()}


@app.get("/api/samples", tags=["system"])
def samples():
    """Bundled demo media.

    The frontend fetches one of these as a blob and posts it exactly as if the
    operator had chosen it from a file dialog, so samples and real uploads
    travel the same code path.
    """
    presets = {
        "street-bus.jpg": {"label": "City sidewalk - bus and pedestrians",
                           "preset": "street", "kind": "image"},
        "street-pedestrians.jpg": {"label": "Street - pedestrians",
                                   "preset": "street", "kind": "image"},
        "street-pan.mp4": {"label": "Street pan (synthetic camera move)",
                           "preset": "street", "kind": "video"},
    }

    out = []
    for path in sorted(config.SAMPLE_DIR.glob("*")):
        suffix = path.suffix.lower()
        if suffix in config.ALLOWED_SUFFIXES:
            kind = "image"
        elif suffix in config.ALLOWED_VIDEO_SUFFIXES:
            kind = "video"
        else:
            continue

        meta = presets.get(
            path.name,
            {"label": path.stem.replace("-", " ").title(),
             "preset": "drone_oblique", "kind": kind},
        )
        meta["kind"] = kind
        out.append({
            "filename": path.name,
            "url": f"/samples/{path.name}",
            "size_bytes": path.stat().st_size,
            **meta,
        })
    return {"samples": out}


# --------------------------------------------------------------------------
# Image pipeline
# --------------------------------------------------------------------------


@app.post("/api/analyze", response_model=DetectResponse, tags=["pipeline"])
async def analyze(
    file: UploadFile = File(..., description="Image to analyse"),
    hfov_deg: float = Form(78.0),
    altitude_m: float = Form(60.0),
    pitch_deg: float = Form(45.0),
    heading_deg: float = Form(0.0),
    roll_deg: float = Form(0.0),
    lat: float = Form(47.3769),
    lon: float = Form(8.5417),
    max_range_m: float = Form(500.0),
    confidence: float = Form(0.35),
    iou: float = Form(0.45),
    max_detections: int = Form(100),
):
    """Run the whole pipeline on one uploaded image."""
    t_start = time.perf_counter()

    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in config.ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: "
            + ", ".join(sorted(config.ALLOWED_SUFFIXES)),
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(raw) / 1e6:.1f} MB, limit is "
                   f"{config.MAX_UPLOAD_BYTES / 1e6:.0f} MB.",
        )

    try:
        cam_params = CameraParams(
            hfov_deg=hfov_deg, altitude_m=altitude_m, pitch_deg=pitch_deg,
            heading_deg=heading_deg % 360.0, roll_deg=roll_deg,
            lat=lat, lon=lon, max_range_m=max_range_m,
        )
        det_params = DetectParams(
            confidence=confidence, iou=iou, max_detections=max_detections
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # --- stage 1: decode and normalise -------------------------------------
    array, pil_image = _load_image(raw, file.filename or "upload")
    image_id = uuid.uuid4().hex[:16]
    stored_name = f"{image_id}{'.png' if suffix == '.png' else '.jpg'}"
    stored_path = config.UPLOAD_DIR / stored_name
    pil_image.save(stored_path, quality=92)

    exif_hints = extract_camera_hints(str(stored_path), pil_image.width)

    # --- stage 2: detection -------------------------------------------------
    try:
        result = detector.detect(
            array,
            confidence=det_params.confidence,
            iou=det_params.iou,
            max_detections=det_params.max_detections,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Detection failed")
        raise HTTPException(status_code=503, detail=f"Detection failed: {exc}") from exc

    # --- stage 3: geometry --------------------------------------------------
    cam = _build_camera(cam_params, result.image_width, result.image_height)
    detections = _georeference(result.detections, cam)
    footprint = [FootprintPoint(**p) for p in ground_footprint(cam)]

    _cache_put(image_id, {
        "result": result,
        "path": str(stored_path),
        "filename": file.filename or stored_name,
        "url": f"/uploads/{stored_name}",
        "size_bytes": len(raw),
        "exif": exif_hints,
    })

    total_ms = (time.perf_counter() - t_start) * 1000.0
    log.info("Analysed %s: %d detections, %d located, %.0f ms inference",
             file.filename, len(detections),
             sum(1 for d in detections if d.world.valid), result.inference_ms)

    response = DetectResponse(
        image=ImageInfo(
            width=result.image_width, height=result.image_height,
            filename=file.filename or stored_name,
            url=f"/uploads/{stored_name}", size_bytes=len(raw),
        ),
        camera=_camera_out(cam),
        detections=detections,
        footprint=footprint,
        model_name=result.model_name,
        inference_ms=round(result.inference_ms, 1),
        total_ms=round(total_ms, 1),
        georeferenced_count=sum(1 for d in detections if d.world.valid),
    )

    payload = json.loads(response.model_dump_json())
    payload["image_id"] = image_id
    payload["exif"] = exif_hints
    payload["media_kind"] = "image"
    payload["metrics"] = analytics.summarise(
        payload["detections"], payload["footprint"],
        result.inference_ms, total_ms,
    )
    payload["scene"] = _scene_payload(payload["detections"], payload["camera"])
    return JSONResponse(payload)


class ReprojectRequest(BaseModel):
    image_id: str
    camera: CameraParams


@app.post("/api/reproject", response_model=DetectResponse, tags=["pipeline"])
def reproject(req: ReprojectRequest):
    """Redo only the geometry stage, reusing the cached detections."""
    t_start = time.perf_counter()
    entry = _cache_get(req.image_id)
    result: DetectionResult = entry["result"]

    cam = _build_camera(req.camera, result.image_width, result.image_height)
    detections = _georeference(result.detections, cam)
    footprint = [FootprintPoint(**p) for p in ground_footprint(cam)]

    total_ms = (time.perf_counter() - t_start) * 1000.0
    response = DetectResponse(
        image=ImageInfo(
            width=result.image_width, height=result.image_height,
            filename=entry["filename"], url=entry["url"],
            size_bytes=entry["size_bytes"],
        ),
        camera=_camera_out(cam),
        detections=detections,
        footprint=footprint,
        model_name=result.model_name,
        inference_ms=0.0,  # the detector was not re-run
        total_ms=round(total_ms, 1),
        georeferenced_count=sum(1 for d in detections if d.world.valid),
    )

    payload = json.loads(response.model_dump_json())
    payload["image_id"] = req.image_id
    payload["exif"] = entry.get("exif", {})
    payload["media_kind"] = "image"
    payload["metrics"] = analytics.summarise(
        payload["detections"], payload["footprint"], 0.0, total_ms,
    )
    payload["scene"] = _scene_payload(payload["detections"], payload["camera"])
    return JSONResponse(payload)


# --------------------------------------------------------------------------
# Scene graph
# --------------------------------------------------------------------------


class ScenePayload(BaseModel):
    """A set of already geo-referenced detections plus the camera that made them.

    Passed straight back from the client so the same endpoint serves a still
    image, one frame of a video, or a filtered subset -- whatever is on screen.
    """

    camera: dict = Field(..., description="A camera object as returned by /api/analyze")
    detections: list[dict] = Field(default_factory=list)


@app.post("/api/scene", tags=["scene"])
def scene(payload: ScenePayload):
    """Spatial relationships for an arbitrary set of detections.

    Pure arithmetic over positions the geometry stage already produced, so this
    is fast enough to call as the operator scrubs through a video.
    """
    return _scene_payload(payload.detections, payload.camera)


# --------------------------------------------------------------------------
# Video pipeline
# --------------------------------------------------------------------------


@app.post("/api/video/analyze", tags=["video"])
async def video_analyze(
    file: UploadFile = File(..., description="Video to track"),
    confidence: float = Form(0.35),
    iou: float = Form(0.45),
    target_fps: float = Form(video_mod.DEFAULT_TARGET_FPS),
):
    """Start tracking a video. Returns a job id to poll.

    Tracking a minute of footage is hundreds of inferences, far too long to hold
    an HTTP request open, so the work runs in the background and the client
    polls /api/video/job/{id}.
    """
    suffix = Path(file.filename or "clip.mp4").suffix.lower()
    if suffix not in config.ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{suffix}'. Allowed: "
            + ", ".join(sorted(config.ALLOWED_VIDEO_SUFFIXES)),
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > config.MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Video is {len(raw) / 1e6:.0f} MB, limit is "
                   f"{config.MAX_VIDEO_BYTES / 1e6:.0f} MB.",
        )

    video_id = uuid.uuid4().hex[:16]
    stored_name = f"{video_id}{suffix}"
    stored_path = config.UPLOAD_DIR / stored_name
    stored_path.write_bytes(raw)

    job = video_processor.submit(
        stored_path,
        filename=file.filename or stored_name,
        url=f"/uploads/{stored_name}",
        confidence=confidence,
        iou=iou,
        target_fps=target_fps,
    )
    log.info("Video job %s queued for %s (%.1f MB)",
             job.id, file.filename, len(raw) / 1e6)
    return job.public()


@app.get("/api/video/job/{job_id}", tags=["video"])
def video_job(job_id: str):
    job = video_processor.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown video job id.")
    return job.public()


# --------------------------------------------------------------------------
# KITTI: real sensor data
# --------------------------------------------------------------------------


@app.get("/api/kitti/sequences", tags=["kitti"])
def kitti_sequences():
    """Real KITTI raw drives available under data/kitti."""
    try:
        found = kitti.discover(config.KITTI_DIR)
    except Exception as exc:  # noqa: BLE001
        log.warning("KITTI discovery failed: %s", exc)
        return {"sequences": [], "error": str(exc)}
    return {"sequences": [s.summary() for s in found]}


class KittiRequest(BaseModel):
    sequence: str
    confidence: float = Field(0.35, ge=0.01, le=0.99)
    iou: float = Field(0.45, ge=0.1, le=0.95)


@app.post("/api/kitti/analyze", tags=["kitti"])
def kitti_analyze(req: KittiRequest):
    """Fuse YOLO detections with the drive's real LiDAR, GPS/IMU and calibration.

    Returns a job id to poll, because a sequence is one inference plus one
    120k-point cloud projection per frame.
    """
    sequence = kitti.find_sequence(config.KITTI_DIR, req.sequence)
    if sequence is None:
        raise HTTPException(status_code=404, detail=f"Unknown KITTI sequence '{req.sequence}'.")

    def worker(job):
        return kitti.process_sequence(
            sequence, detector, config.UPLOAD_DIR,
            confidence=req.confidence, iou=req.iou, job=job,
        )

    job = video_processor.submit_worker(
        worker, filename=sequence.name, url="", path="",
    )
    log.info("KITTI job %s queued for %s (%d frames)",
             job.id, sequence.name, sequence.frame_count)
    return job.public()


class KittiTimelineRequest(BaseModel):
    job_id: str
    # The camera pose comes from the sensors, so there is nothing to supply.
    # max_range_m only bounds the ground-plane fallback.
    max_range_m: float = Field(120.0, gt=1.0, le=20000.0)


@app.post("/api/kitti/timeline", tags=["kitti"])
def kitti_timeline(req: KittiTimelineRequest):
    """Turn a processed KITTI job into the timeline the viewer consumes.

    Each detection is placed one of two ways, and which one is recorded:

      * **measured** -- the median of the LiDAR returns that landed inside its
        box, rotated into world coordinates by the GPS/IMU orientation. Real
        depth from a real sensor.
      * **estimated** -- no usable LiDAR return, so it falls back to the
        ground-plane projection in geo.py, with the camera pose derived from
        the calibration and the measured road surface.

    The vehicle is moving, so every frame carries its own camera pose.
    """
    job = video_processor.get(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown KITTI job id.")
    if job.status != "done" or job.result is None:
        raise HTTPException(
            status_code=409,
            detail=f"KITTI job is '{job.status}', not ready yet.",
        )

    t_start = time.perf_counter()
    result: kitti.KittiResult = job.result

    frames: list[dict] = []
    measured_total = 0
    estimated_total = 0

    for frame in result.frames:
        pose = frame.camera
        cam = CameraModel(
            image_width=result.width,
            image_height=result.height,
            hfov_deg=pose["hfov_deg"],
            altitude_m=pose["altitude_m"],
            pitch_deg=pose["pitch_deg"],
            heading_deg=pose["heading_deg"],
            lat=pose["lat"], lon=pose["lon"],
            max_range_m=req.max_range_m,
            fx_px=pose["fx_px"], fy_px=pose["fy_px"],
            cx_px=pose["cx_px"], cy_px=pose["cy_px"],
        )

        out: list[dict] = []
        for det, fix in zip(frame.detections, frame.fixes):
            if fix.ok:
                world = _world_from_lidar(cam, fix)
                measured_total += 1
            else:
                world = _world_from_ground_plane(cam, det, fix.reason)
                if world.valid:
                    estimated_total += 1
            out.append(json.loads(_detection_out(det, world).model_dump_json()))

        frames.append({
            "index": frame.index,
            "time_s": round(frame.time_s, 3),
            "detections": out,
            "camera": json.loads(_camera_out(cam).model_dump_json()),
            "speed_kmh": round(frame.speed_kmh, 1),
            "lidar_points": frame.lidar_points,
        })

    trails = video_mod.build_trails(frames)

    first_cam = frames[0]["camera"]
    footprint = []  # the ego camera is level and forward-facing; see note below
    all_detections = [d for f in frames for d in f["detections"]]
    metrics = analytics.summarise(
        all_detections, footprint,
        inference_ms=result.processing_ms / max(1, result.frame_count),
        total_ms=(time.perf_counter() - t_start) * 1000.0,
        frame_count=result.frame_count,
        video_ms=result.processing_ms,
    )
    metrics["track_count"] = len(trails)
    metrics["objects_per_frame"] = round(len(all_detections) / max(1, len(frames)), 1)
    metrics["measured_count"] = measured_total
    metrics["estimated_count"] = estimated_total

    return {
        "media_kind": "kitti",
        "job_id": job.id,
        "sequence": result.sequence,
        "calibration": result.calibration,
        "video": {
            "url": job.url,
            "filename": result.sequence,
            "width": result.width,
            "height": result.height,
            "duration_s": round(result.duration_s, 2),
            "source_fps": result.source_fps,
            "processed_fps": result.target_fps,
            "frame_count": result.frame_count,
        },
        "camera": first_cam,
        "footprint": footprint,
        "frames": frames,
        "trails": trails,
        "metrics": metrics,
        "model_name": config.MODEL_WEIGHTS,
        "sensors": {
            "camera": "Point Grey Flea 2 colour, rectified 1242x375",
            "lidar": "Velodyne HDL-64E, ~120k points/frame",
            "gnss_imu": "OXTS RT3003 (lat/lon/alt + roll/pitch/yaw)",
        },
    }


def _world_from_lidar(cam: CameraModel, fix: kitti.LidarFix) -> WorldPosition:
    """A measured position: straight from the LiDAR returns."""
    lat, lon = cam.enu_to_latlon(fix.east_m, fix.north_m)
    bearing = math.degrees(math.atan2(fix.east_m, fix.north_m)) % 360.0
    return WorldPosition(
        valid=True,
        source="lidar",
        lidar_points=fix.point_count,
        depth_m=round(fix.depth_m, 2),
        east_m=round(fix.east_m, 3),
        north_m=round(fix.north_m, 3),
        up_m=round(fix.up_m, 3),
        lat=lat, lon=lon,
        ground_range_m=round(fix.distance_m, 2),
        slant_range_m=round(math.sqrt(fix.distance_m ** 2 + fix.up_m ** 2), 2),
        bearing_deg=round(bearing, 1),
        est_width_m=round(fix.width_m, 2),
        est_height_m=round(fix.height_m, 2),
    )


def _world_from_ground_plane(cam: CameraModel, det: Detection, why: str) -> WorldPosition:
    """The fallback: no LiDAR return, so project onto the ground plane."""
    fix = project_detection(cam, det.x1, det.y1, det.x2, det.y2)
    reason = fix.reason
    if not fix.valid and why:
        reason = f"{why}; {fix.reason}"
    return WorldPosition(
        valid=fix.valid,
        reason=reason,
        source="ground_plane",
        east_m=round(fix.east_m, 3),
        north_m=round(fix.north_m, 3),
        lat=fix.lat, lon=fix.lon,
        ground_range_m=round(fix.ground_range_m, 2),
        slant_range_m=round(fix.slant_range_m, 2),
        bearing_deg=round(fix.bearing_deg, 1),
        est_width_m=round(fix.width_m, 2),
        est_height_m=round(fix.height_m, 2),
    )


def _detection_out(det: Detection, world: WorldPosition) -> DetectionOut:
    cx, cy = det.center
    ax, ay = det.ground_anchor
    return DetectionOut(
        id=det.id,
        track_id=det.track_id,
        class_id=det.class_id,
        class_name=det.class_name,
        confidence=round(det.confidence, 4),
        is_ground_class=det.is_ground_class,
        bbox=BBox(
            x1=round(det.x1, 1), y1=round(det.y1, 1),
            x2=round(det.x2, 1), y2=round(det.y2, 1),
            width=round(det.width, 1), height=round(det.height, 1),
            cx=round(cx, 1), cy=round(cy, 1),
            anchor_x=round(ax, 1), anchor_y=round(ay, 1),
        ),
        world=world,
    )


class VideoTimelineRequest(BaseModel):
    job_id: str
    camera: CameraParams


@app.post("/api/video/timeline", tags=["video"])
def video_timeline(req: VideoTimelineRequest):
    """Geo-reference every tracked frame against a camera pose.

    The tracking ran once. This is the geometry stage applied across the whole
    timeline, so changing the camera re-projects the entire video in
    milliseconds without re-running the detector.
    """
    job = video_processor.get(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown video job id.")
    if job.status != "done" or job.result is None:
        raise HTTPException(
            status_code=409,
            detail=f"Video job is '{job.status}', not ready for projection yet.",
        )

    t_start = time.perf_counter()
    result = job.result
    cam = _build_camera(req.camera, result.width, result.height)
    footprint = [p for p in ground_footprint(cam)]

    frames: list[dict] = []
    total_detections = 0
    for frame in result.frames:
        located = _georeference(frame.detections, cam)
        total_detections += len(located)
        frames.append({
            "index": frame.index,
            "time_s": round(frame.time_s, 3),
            "detections": [json.loads(d.model_dump_json()) for d in located],
        })

    trails = video_mod.build_trails(frames)

    # Metrics across the whole clip, plus the per-frame average.
    all_detections = [d for f in frames for d in f["detections"]]
    metrics = analytics.summarise(
        all_detections, footprint,
        inference_ms=result.processing_ms / max(1, result.frame_count),
        total_ms=(time.perf_counter() - t_start) * 1000.0,
        frame_count=result.frame_count,
        video_ms=result.processing_ms,
    )
    metrics["track_count"] = len(trails)
    metrics["objects_per_frame"] = round(total_detections / max(1, len(frames)), 1)

    return {
        "media_kind": "video",
        "job_id": job.id,
        "video": {
            "url": job.url,
            "filename": job.filename,
            "width": result.width,
            "height": result.height,
            "duration_s": round(result.duration_s, 2),
            "source_fps": round(result.source_fps, 2),
            "processed_fps": round(result.target_fps, 2),
            "frame_count": result.frame_count,
        },
        "camera": json.loads(_camera_out(cam).model_dump_json()),
        "footprint": footprint,
        "frames": frames,
        "trails": trails,
        "metrics": metrics,
        "model_name": config.MODEL_WEIGHTS,
    }


# --------------------------------------------------------------------------
# Assistant
# --------------------------------------------------------------------------


class AssistantRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    camera: dict
    detections: list[dict] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)


@app.post("/api/assistant", tags=["assistant"])
def assistant(req: AssistantRequest):
    """Answer a question about the current scene, and optionally act on it.

    The client sends whatever is currently on screen -- a still image's
    detections, or one frame of a video. The server builds the scene graph and
    hands the language model measurements, never pixels. Tool calls come back as
    actions the frontend applies to every view at once.
    """
    graph = sg.build(req.detections, req.camera)
    reply = llm.ask(req.question, graph, history=req.history)
    reply["scene_tree"] = graph.to_tree(limit=25)
    return reply


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


class ExportRequest(BaseModel):
    """What to export.

    The client sends the current view's data, so an export always matches what
    is on screen -- including any filter or assistant selection.
    """

    camera: dict
    detections: list[dict] = Field(default_factory=list)
    footprint: list[dict] = Field(default_factory=list)
    trails: list[dict] = Field(default_factory=list)
    source: str = "scene"
    metrics: dict = Field(default_factory=dict)
    scene: dict = Field(default_factory=dict)


@app.post("/api/export/{fmt}", tags=["export"])
def export(fmt: str, req: ExportRequest):
    """Export detections as GeoJSON, JSON or CSV."""
    fmt = fmt.lower()
    if fmt not in ("geojson", "json", "csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown export format '{fmt}'. Use geojson, json or csv.",
        )

    filename = exports.filename_for(req.source, fmt)
    disposition = {"Content-Disposition": f'attachment; filename="{filename}"'}

    if fmt == "csv":
        return PlainTextResponse(
            exports.to_csv(req.detections, req.source),
            media_type="text/csv",
            headers=disposition,
        )

    if fmt == "geojson":
        body = exports.to_geojson(
            req.detections, req.camera, req.footprint, req.source, req.trails
        )
    else:
        body = exports.to_json(
            req.detections, req.camera, req.footprint, req.source,
            metrics=req.metrics, scene=req.scene, trails=req.trails,
        )

    return JSONResponse(body, headers=disposition)


# --------------------------------------------------------------------------
# Static files
# --------------------------------------------------------------------------

app.mount("/uploads", StaticFiles(directory=config.UPLOAD_DIR), name="uploads")
app.mount("/samples", StaticFiles(directory=config.SAMPLE_DIR), name="samples")


@app.get("/", include_in_schema=False)
def index():
    index_file = config.FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            {"detail": "Frontend not found. Expected frontend/index.html."},
            status_code=500,
        )
    return FileResponse(index_file)


# Mounted last so it does not shadow the API routes above.
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
