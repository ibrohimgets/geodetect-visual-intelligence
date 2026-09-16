"""
Pydantic models describing the JSON that flows between backend and frontend.

Keeping these in one file means the API contract is readable in a single sitting,
and FastAPI turns them into interactive docs at /docs for free.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CameraParams(BaseModel):
    """Camera pose and optics, sent up with the image.

    These are the numbers a real survey would read off the drone telemetry or
    the camera installation record. Here the operator types them in, and the
    defaults describe a typical oblique drone shot.
    """

    hfov_deg: float = Field(78.0, gt=1.0, lt=179.0, description="Horizontal field of view")
    altitude_m: float = Field(60.0, gt=0.0, le=10000.0, description="Camera height above ground")
    pitch_deg: float = Field(45.0, ge=0.0, le=90.0, description="Tilt down from horizontal; 90 = straight down")
    heading_deg: float = Field(0.0, ge=0.0, lt=360.0, description="Compass bearing of the lens")
    roll_deg: float = Field(0.0, ge=-45.0, le=45.0, description="Rotation about the lens axis")
    lat: float = Field(47.3769, ge=-90.0, le=90.0, description="Camera latitude, WGS84")
    lon: float = Field(8.5417, ge=-180.0, le=180.0, description="Camera longitude, WGS84")
    max_range_m: float = Field(500.0, gt=1.0, le=20000.0, description="Discard ground hits beyond this")


class DetectParams(BaseModel):
    """Inference knobs."""

    confidence: float = Field(0.35, ge=0.01, le=0.99)
    iou: float = Field(0.45, ge=0.1, le=0.95)
    max_detections: int = Field(100, ge=1, le=300)


class BBox(BaseModel):
    """Bounding box in pixels, top-left origin."""

    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    cx: float = Field(..., description="Centre x, pixels")
    cy: float = Field(..., description="Centre y, pixels")
    anchor_x: float = Field(..., description="Ground contact point x, pixels")
    anchor_y: float = Field(..., description="Ground contact point y, pixels")


class WorldPosition(BaseModel):
    """Where the object ended up in the real world."""

    valid: bool = Field(..., description="False when no position could be derived")
    reason: str = ""
    source: str = Field(
        "ground_plane",
        description=(
            "How the position was obtained. 'lidar' means it was MEASURED from "
            "real depth returns; 'ground_plane' means it was ESTIMATED by "
            "projecting onto an assumed flat ground. They are not equivalent and "
            "the UI labels them differently."
        ),
    )
    lidar_points: int = Field(
        0, description="Number of LiDAR returns used, when source is 'lidar'"
    )
    depth_m: float = Field(
        0.0, description="Range along the camera's optical axis, when measured"
    )
    east_m: float = 0.0
    north_m: float = 0.0
    up_m: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    ground_range_m: float = 0.0
    slant_range_m: float = 0.0
    bearing_deg: float = 0.0
    est_width_m: float = 0.0
    est_height_m: float = 0.0


class DetectionOut(BaseModel):
    """One detection, carried all the way through the pipeline."""

    id: int
    track_id: int | None = Field(
        None,
        description="Video mode only: an identity that stays with the same "
        "physical object across frames, which is what motion trails are built from",
    )
    class_id: int
    class_name: str
    confidence: float
    is_ground_class: bool = Field(
        ...,
        description="True when this class normally rests on the ground, so the "
        "flat-ground assumption is reasonable for it",
    )
    bbox: BBox
    world: WorldPosition


class FootprintPoint(BaseModel):
    lat: float
    lon: float
    east_m: float
    north_m: float


class CameraOut(BaseModel):
    """Echo of the camera model, including values the backend derived."""

    lat: float
    lon: float
    altitude_m: float
    pitch_deg: float
    heading_deg: float
    roll_deg: float
    hfov_deg: float
    vfov_deg: float
    fx: float
    fy: float
    max_range_m: float


class ImageInfo(BaseModel):
    width: int
    height: int
    filename: str
    url: str
    size_bytes: int


class DetectResponse(BaseModel):
    """The single payload the frontend needs to draw all three views."""

    ok: bool = True
    image: ImageInfo
    camera: CameraOut
    detections: list[DetectionOut]
    footprint: list[FootprintPoint] = Field(
        default_factory=list,
        description="Image corners projected onto the ground -- the camera coverage polygon",
    )
    model_name: str
    inference_ms: float
    total_ms: float
    georeferenced_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    num_classes: int = 0
    error: str | None = None
    assistant: dict = Field(
        default_factory=dict,
        description="Whether the language assistant is configured, and which model it uses",
    )


class ClassesResponse(BaseModel):
    classes: dict[int, str]
    ground_classes: list[str]
