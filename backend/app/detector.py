"""
Object detection using a pretrained YOLOv8 model.

Nothing is trained here. We load Ultralytics YOLOv8, which ships pretrained on
the COCO dataset (80 everyday classes -- person, car, truck, traffic light,
bench and so on), and run inference on the uploaded image.

The model file is downloaded automatically on first use and cached in the
project models/ folder, so the first run needs a network connection and later
runs do not.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("cadsoftware.detector")

# Classes that generally sit ON the ground, so the ground-plane assumption in
# geo.py holds for them. Anything else still gets detected and drawn, it is
# just flagged so the operator knows the position is less trustworthy.
GROUND_CLASSES = {
    # people and vehicles
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "boat",
    # street furniture
    "bench", "fire hydrant", "stop sign", "parking meter", "traffic light",
    # animals
    "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # things that sit on the ground in an outdoor scene
    "potted plant", "chair", "couch", "dining table", "suitcase",
    "skateboard", "skis", "snowboard", "sports ball",
}


@dataclass
class Detection:
    """One detected object, in image space only. Geo-referencing happens later."""

    id: int
    class_id: int
    class_name: str
    confidence: float
    # Bounding box in pixels, top-left origin.
    x1: float
    y1: float
    x2: float
    y2: float
    # Set only in video mode: a tracker id that stays with the same physical
    # object across frames, which is what turns a pile of per-frame boxes into
    # something you can draw a motion trail for.
    track_id: Optional[int] = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def ground_anchor(self) -> tuple[float, float]:
        """Bottom-centre of the box -- where we assume the object meets the ground."""
        return (self.x1 + self.x2) / 2.0, self.y2

    @property
    def is_ground_class(self) -> bool:
        return self.class_name in GROUND_CLASSES


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    model_name: str = ""
    image_width: int = 0
    image_height: int = 0


class Detector:
    """Thin wrapper around Ultralytics YOLO.

    Loading a model takes a second or two, so we do it once, lazily, and keep it
    in memory. A lock guards both the load and inference itself, because the
    underlying model is not safe to call from several threads at once and
    FastAPI will happily hand us concurrent requests.
    """

    def __init__(self, weights: str = "yolov8n.pt", models_dir: Optional[Path] = None):
        self.weights = weights
        self.models_dir = models_dir
        self._model = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ loading

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def resolve_weights(self) -> str:
        """Where to load the checkpoint from.

        Ultralytics downloads into the current working directory, so prefer our
        cached copy in models/ when it is already there.
        """
        if self.models_dir:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            local = self.models_dir / self.weights
            if local.exists():
                return str(local)
        return self.weights

    def new_tracking_model(self):
        """A private YOLO instance for one video job.

        Ultralytics keeps tracker state on the model object when `persist=True`,
        so two videos sharing one model would contaminate each other's track
        ids. Loading a second instance costs a fraction of a second and keeps
        each job's identities its own.
        """
        from ultralytics import YOLO

        return YOLO(self.resolve_weights())

    def load(self):
        """Load the model, downloading the weights on first use."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:  # another thread beat us to it
                return self._model
            try:
                from ultralytics import YOLO

                path = self.resolve_weights()
                log.info("Loading detection model: %s", path)
                t0 = time.perf_counter()
                self._model = YOLO(path)
                log.info("Model ready in %.2fs", time.perf_counter() - t0)
                self._load_error = None
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                self._load_error = str(exc)
                log.exception("Model failed to load")
                raise

        return self._model

    @property
    def class_names(self) -> dict[int, str]:
        model = self.load()
        return dict(model.names)

    # ---------------------------------------------------------------- inference

    def detect(
        self,
        image: np.ndarray,
        confidence: float = 0.35,
        iou: float = 0.45,
        max_detections: int = 100,
        classes: Optional[list[int]] = None,
    ) -> DetectionResult:
        """Run detection on an RGB image array.

        confidence     drop boxes the model is less sure about than this
        iou            IoU threshold for non-maximum suppression, which removes
                       duplicate boxes covering the same object
        max_detections cap on how many boxes come back
        classes        optional whitelist of COCO class ids
        """
        model = self.load()
        h, w = image.shape[:2]

        t0 = time.perf_counter()
        with self._lock:
            results = model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                max_det=max_detections,
                classes=classes,
                verbose=False,
            )
        elapsed = (time.perf_counter() - t0) * 1000.0

        detections: list[Detection] = []
        if results:
            boxes = results[0].boxes
            names = results[0].names
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                # Highest confidence first, so the results panel is ordered the
                # way an operator would want to read it.
                order = np.argsort(-confs)
                for new_id, i in enumerate(order):
                    x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                    detections.append(
                        Detection(
                            id=new_id,
                            class_id=int(clss[i]),
                            class_name=str(names[int(clss[i])]),
                            confidence=float(confs[i]),
                            # Clamp to the image, as boxes can spill over the edge.
                            x1=max(0.0, x1),
                            y1=max(0.0, y1),
                            x2=min(float(w), x2),
                            y2=min(float(h), y2),
                        )
                    )

        return DetectionResult(
            detections=detections,
            inference_ms=elapsed,
            model_name=self.weights,
            image_width=w,
            image_height=h,
        )
