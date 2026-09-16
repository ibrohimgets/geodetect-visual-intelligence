"""
Scene metrics for the dashboard tiles.

Small, cheap summaries computed from what the pipeline already produced. No
extra inference, no extra geometry -- just arithmetic over the detections and
the camera footprint.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .categories import category_of


def footprint_area_m2(footprint: list[dict]) -> Optional[float]:
    """Ground area covered by the photo, in square metres.

    The shoelace formula over the footprint polygon in local ENU metres. Returns
    None when the footprint is not a closed polygon, which happens on an oblique
    shot whose upper corners are sky -- in that case the visible area genuinely
    is unbounded, and reporting a number would be a lie.
    """
    if not footprint or len(footprint) < 3:
        return None

    area = 0.0
    n = len(footprint)
    for i in range(n):
        a = footprint[i]
        b = footprint[(i + 1) % n]
        area += a["east_m"] * b["north_m"] - b["east_m"] * a["north_m"]
    return abs(area) / 2.0


def summarise(
    detections: Iterable[dict],
    footprint: list[dict],
    inference_ms: float,
    total_ms: float,
    frame_count: int = 1,
    video_ms: Optional[float] = None,
) -> dict[str, Any]:
    """Everything the analytics strip displays."""
    detections = list(detections)
    located = [d for d in detections if d.get("world", {}).get("valid")]

    confidences = [float(d["confidence"]) for d in detections]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    classes: dict[str, int] = {}
    categories: dict[str, int] = {}
    for det in detections:
        name = det["class_name"]
        classes[name] = classes.get(name, 0) + 1
        cat = category_of(name)
        categories[cat] = categories.get(cat, 0) + 1

    nearest = farthest = None
    if located:
        by_range = sorted(located, key=lambda d: d["world"]["ground_range_m"])
        nearest = _brief(by_range[0])
        farthest = _brief(by_range[-1])

    area = footprint_area_m2(footprint)

    # For video, frames per second of actual processing. For a still image the
    # equivalent figure is how many such frames the detector could sustain.
    fps = None
    if video_ms and video_ms > 0 and frame_count > 0:
        fps = frame_count / (video_ms / 1000.0)
    elif inference_ms > 0:
        fps = 1000.0 / inference_ms

    return {
        "object_count": len(detections),
        "located_count": len(located),
        "class_count": len(classes),
        "classes": classes,
        "categories": categories,
        "avg_confidence": round(avg_conf, 3),
        "inference_ms": round(inference_ms, 1),
        "total_ms": round(total_ms, 1),
        "fps": round(fps, 1) if fps else None,
        "frame_count": frame_count,
        "nearest": nearest,
        "farthest": farthest,
        "visible_area_m2": round(area, 1) if area is not None else None,
        "visible_area_note": (
            None if area is not None
            else "open footprint - the view reaches the horizon, so the area is unbounded"
        ),
    }


def _brief(det: dict) -> dict[str, Any]:
    return {
        "id": det["id"],
        "class": det["class_name"],
        "distance_m": round(det["world"]["ground_range_m"], 1),
    }
