"""
Export the detection results in the formats other tools actually read.

    geojson  drop straight into QGIS or ArcGIS
    json     the full record, including camera pose and scene graph
    csv      open in a spreadsheet

The GeoJSON writer is the one with a trap in it: GeoJSON coordinates are always
[longitude, latitude] in that order, and polygons must repeat their first point
to close the ring. Both are easy to get backwards and neither fails loudly.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Optional

from .categories import category_of


def to_geojson(
    detections: list[dict],
    camera: dict,
    footprint: list[dict],
    source: str,
    trails: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """A FeatureCollection holding the detections, the camera and its footprint."""
    features: list[dict[str, Any]] = []

    for det in detections:
        world = det.get("world", {})
        if not world.get("valid"):
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                # GeoJSON is longitude first. This is the classic mistake.
                "type": "Point",
                "coordinates": [world["lon"], world["lat"]],
            },
            "properties": {
                "id": det["id"],
                "track_id": det.get("track_id"),
                "class": det["class_name"],
                "class_id": det["class_id"],
                "category": category_of(det["class_name"]),
                "confidence": det["confidence"],
                "east_m": world["east_m"],
                "north_m": world["north_m"],
                "distance_m": world["ground_range_m"],
                "bearing_deg": world["bearing_deg"],
                "est_width_m": world["est_width_m"],
                "est_height_m": world["est_height_m"],
                "pixel_x": det["bbox"]["cx"],
                "pixel_y": det["bbox"]["cy"],
                "position_type": "estimated",
                "source_image": source,
            },
        })

    if len(footprint) >= 3:
        ring = [[p["lon"], p["lat"]] for p in footprint]
        ring.append(ring[0])  # a GeoJSON ring must close
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"role": "camera_footprint"},
        })

    for trail in trails or []:
        points = trail.get("points", [])
        if len(points) < 2:
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[p["lon"], p["lat"]] for p in points],
            },
            "properties": {
                "role": "track",
                "track_id": trail["track_id"],
                "label": trail["label"],
                "class": trail["class_name"],
                "path_length_m": trail.get("distance_m"),
            },
        })

    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [camera["lon"], camera["lat"]]},
        "properties": {
            "role": "camera",
            "altitude_m": camera["altitude_m"],
            "pitch_deg": camera["pitch_deg"],
            "heading_deg": camera["heading_deg"],
            "hfov_deg": camera["hfov_deg"],
        },
    })

    return {
        "type": "FeatureCollection",
        "name": "geodetect_detections",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }


def to_json(
    detections: list[dict],
    camera: dict,
    footprint: list[dict],
    source: str,
    metrics: Optional[dict] = None,
    scene: Optional[dict] = None,
    trails: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """The complete record: everything the pipeline knows, in one document."""
    return {
        "generator": "GeoDetect",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "position_note": (
            "Positions are ESTIMATED by projecting monocular detections onto an "
            "assumed flat ground plane. They are not survey measurements."
        ),
        "camera": camera,
        "footprint": footprint,
        "metrics": metrics or {},
        "scene_graph": scene or {},
        "trails": trails or [],
        "detections": detections,
    }


CSV_COLUMNS = [
    "id", "track_id", "class", "category", "confidence",
    "pixel_x", "pixel_y", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "latitude", "longitude", "east_m", "north_m",
    "distance_m", "bearing_deg", "est_width_m", "est_height_m",
    "position_estimated", "source",
]


def to_csv(detections: list[dict], source: str) -> str:
    """One row per detection, including those with no ground fix.

    Objects without a position keep their row -- they were genuinely detected --
    but their spatial columns are left empty rather than filled with zeros,
    which would read as "at the camera".
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()

    for det in detections:
        world = det.get("world", {})
        bbox = det["bbox"]
        located = bool(world.get("valid"))

        writer.writerow({
            "id": det["id"],
            "track_id": det.get("track_id", ""),
            "class": det["class_name"],
            "category": category_of(det["class_name"]),
            "confidence": round(det["confidence"], 4),
            "pixel_x": bbox["cx"],
            "pixel_y": bbox["cy"],
            "bbox_x1": bbox["x1"], "bbox_y1": bbox["y1"],
            "bbox_x2": bbox["x2"], "bbox_y2": bbox["y2"],
            "latitude": round(world["lat"], 7) if located else "",
            "longitude": round(world["lon"], 7) if located else "",
            "east_m": world["east_m"] if located else "",
            "north_m": world["north_m"] if located else "",
            "distance_m": world["ground_range_m"] if located else "",
            "bearing_deg": world["bearing_deg"] if located else "",
            "est_width_m": world["est_width_m"] if located else "",
            "est_height_m": world["est_height_m"] if located else "",
            "position_estimated": "yes" if located else "no ground fix",
            "source": source,
        })

    return buffer.getvalue()


def filename_for(source: str, fmt: str) -> str:
    stem = source.rsplit(".", 1)[0] or "geodetect"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ext = {"geojson": "geojson", "json": "json", "csv": "csv"}[fmt]
    return f"{stem}_detections_{stamp}.{ext}"
