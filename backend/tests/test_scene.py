"""
Tests for the scene-intelligence layer: categories, the spatial scene graph,
the query language the assistant drives, analytics and exports.

    python backend/tests/test_scene.py
    pytest backend/tests/test_scene.py

None of these touch the network or the detector. The scene graph is a pure
function of the geo stage's output, which is exactly what makes it testable:
we can hand it detections at positions we chose and assert on the relationships
it derives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import analytics, exports  # noqa: E402
from app import scene_graph as sg  # noqa: E402
from app.categories import (  # noqa: E402
    category_of,
    classes_in,
    resolve_categories,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def make_detection(det_id, class_name, class_id, east, north, *,
                   confidence=0.9, valid=True, width=1.0, height=1.8,
                   track_id=None):
    """A detection in the shape the geo stage emits."""
    import math

    distance = math.hypot(east, north)
    bearing = math.degrees(math.atan2(east, north)) % 360.0
    return {
        "id": det_id,
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "is_ground_class": True,
        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "width": 10, "height": 10,
                 "cx": 5, "cy": 5, "anchor_x": 5, "anchor_y": 10},
        "world": {
            "valid": valid, "reason": "" if valid else "above horizon",
            "east_m": east, "north_m": north, "up_m": 0.0,
            "lat": 47.3769 + north / 111132.0,
            "lon": 8.5417 + east / 75000.0,
            "ground_range_m": distance, "slant_range_m": distance,
            "bearing_deg": bearing,
            "est_width_m": width, "est_height_m": height,
        },
    }


CAMERA = {
    "lat": 47.3769, "lon": 8.5417, "altitude_m": 2.0,
    "pitch_deg": 10.0, "heading_deg": 0.0, "roll_deg": 0.0,
    "hfov_deg": 70.0, "vfov_deg": 45.0, "fx": 600.0, "fy": 600.0,
    "max_range_m": 300.0,
}


def simple_scene():
    """A bus dead ahead, one person to its left, one to its right, one far off."""
    return [
        make_detection(0, "bus", 5, 0.0, 20.0, width=2.5, height=3.2),
        make_detection(1, "person", 0, -6.0, 20.0),
        make_detection(2, "person", 0, 6.0, 20.0),
        make_detection(3, "person", 0, 2.0, 80.0),
        make_detection(4, "car", 2, 0.0, 0.0, valid=False),
    ]


# --------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------


def test_classes_map_to_sensible_categories():
    assert category_of("person") == "people"
    assert category_of("bus") == "vehicles"
    assert category_of("dog") == "animals"
    assert category_of("chair") == "furniture"
    assert category_of("traffic light") == "outdoor"
    assert category_of("backpack") == "personal items"


def test_unmapped_class_falls_back_to_other():
    assert category_of("definitely not a coco class") == "other"


def test_category_lookup_is_tolerant_of_phrasing():
    # A language model might say any of these.
    for word in ["vehicles", "Vehicles", "vehicle", "cars", "CARS"]:
        assert "bus" in resolve_categories([word]), word
    for word in ["people", "person", "pedestrians", "Humans"]:
        assert "person" in resolve_categories([word]), word


def test_resolve_categories_merges_several():
    result = resolve_categories(["people", "animals"])
    assert "person" in result and "dog" in result
    assert "bus" not in result


def test_unknown_category_resolves_to_nothing():
    assert resolve_categories(["spaceships"]) == set()
    assert classes_in("spaceships") == []


# --------------------------------------------------------------------------
# Scene graph
# --------------------------------------------------------------------------


def test_graph_labels_and_counts():
    graph = sg.build(simple_scene(), CAMERA)
    assert len(graph.nodes) == 5
    assert len(graph.located) == 4  # the invalid one has no position

    labels = [n.label for n in graph.nodes]
    assert labels[0] == "Bus #1"
    assert labels[1] == "Person #1"
    assert labels[2] == "Person #2"  # numbering runs per class


def test_tracked_objects_get_persistent_style_labels():
    detections = [
        make_detection(0, "person", 0, 1.0, 10.0, track_id=7),
        make_detection(1, "car", 2, -3.0, 12.0, track_id=12),
    ]
    graph = sg.build(detections, CAMERA)
    assert graph.nodes[0].label == "person_07"
    assert graph.nodes[1].label == "car_12"


def test_side_is_relative_to_where_the_camera_points():
    graph = sg.build(simple_scene(), CAMERA)
    assert graph.by_id(1).side == "left"    # 6 m west of a north-facing camera
    assert graph.by_id(2).side == "right"
    assert graph.by_id(0).side == "ahead"


def test_side_flips_when_the_camera_turns_around():
    """The same object is on the other hand when the camera faces the other way."""
    south_facing = {**CAMERA, "heading_deg": 180.0}
    graph = sg.build(simple_scene(), south_facing)
    assert graph.by_id(1).side == "right"
    assert graph.by_id(2).side == "left"


def test_distance_bands_are_scene_relative():
    graph = sg.build(simple_scene(), CAMERA)
    # The object 80 m out must be the far one; the three at 20 m are not.
    assert graph.by_id(3).distance_band == "far"
    assert graph.by_id(0).distance_band in ("near", "mid")


def test_unlocated_objects_get_no_spatial_claims():
    graph = sg.build(simple_scene(), CAMERA)
    ghost = graph.by_id(4)
    assert ghost.located is False
    assert ghost.relations == []
    # And the serialised form must not invent coordinates for it.
    payload = ghost.to_dict()
    assert "distance_m" not in payload
    assert "latitude" not in payload


def test_near_relation_uses_a_scene_relative_radius():
    # Two people almost touching, 20 m out.
    detections = [
        make_detection(0, "person", 0, 0.0, 20.0),
        make_detection(1, "person", 0, 0.8, 20.0),
    ]
    graph = sg.build(detections, CAMERA)
    kinds = {r.kind for r in graph.by_id(0).relations}
    assert "near" in kinds


def test_distant_objects_are_not_called_near():
    detections = [
        make_detection(0, "person", 0, -40.0, 20.0),
        make_detection(1, "person", 0, 40.0, 20.0),
    ]
    graph = sg.build(detections, CAMERA)
    kinds = {r.kind for r in graph.by_id(0).relations}
    assert "near" not in kinds
    assert "left_of" in kinds  # it is well to the left of the other


def test_groups_cluster_things_that_are_together():
    detections = [
        make_detection(0, "person", 0, 0.0, 20.0),
        make_detection(1, "person", 0, 1.0, 20.0),
        make_detection(2, "person", 0, 2.0, 20.0),
        make_detection(3, "car", 2, 90.0, 20.0),   # clearly on its own
    ]
    graph = sg.build(detections, CAMERA)
    groups = [g for g in graph.groups if len(g) > 1]
    assert len(groups) == 1
    assert set(groups[0]) == {0, 1, 2}
    assert graph.by_id(3).group_id != graph.by_id(0).group_id


def test_tree_rendering_has_the_expected_shape():
    graph = sg.build(simple_scene(), CAMERA)
    tree = graph.to_tree()
    assert "Bus #1" in tree
    assert "m from camera" in tree
    assert "├─" in tree or "└─" in tree  # box-drawing branches
    # The unlocated object should say so rather than be silently dropped.
    assert "no ground fix" in tree


def test_context_is_json_serialisable_and_summarised():
    graph = sg.build(simple_scene(), CAMERA)
    context = graph.to_context()
    json.dumps(context)  # must not raise

    summary = context["summary"]
    assert summary["object_count"] == 5
    assert summary["located_count"] == 4
    assert summary["class_counts"]["person"] == 3
    assert summary["nearest"]["id"] == 0      # the bus at 20 m
    assert summary["farthest"]["id"] == 3     # the person at 80 m


# --------------------------------------------------------------------------
# Selection -- the operations natural language is translated into
# --------------------------------------------------------------------------


def test_select_by_class():
    graph = sg.build(simple_scene(), CAMERA)
    assert {n.id for n in sg.select(graph, classes=["person"])} == {1, 2, 3}


def test_select_by_category():
    graph = sg.build(simple_scene(), CAMERA)
    assert {n.id for n in sg.select(graph, categories=["vehicles"])} == {0, 4}


def test_select_by_distance():
    graph = sg.build(simple_scene(), CAMERA)
    close = sg.select(graph, max_distance_m=30)
    assert {n.id for n in close} == {0, 1, 2}   # excludes the one at 80 m
    far = sg.select(graph, min_distance_m=50)
    assert {n.id for n in far} == {3}


def test_select_by_side():
    graph = sg.build(simple_scene(), CAMERA)
    assert {n.id for n in sg.select(graph, side="left")} == {1}


def test_select_near_another_object():
    """This is 'people within 25 m of the bus' -- the flagship query."""
    graph = sg.build(simple_scene(), CAMERA)
    near_bus = sg.select(graph, classes=["person"], near_object_id=0, near_radius_m=25)
    # Persons 1 and 2 are 6 m from the bus; person 3 is 60 m away.
    assert {n.id for n in near_bus} == {1, 2}


def test_select_near_excludes_the_anchor_itself():
    graph = sg.build(simple_scene(), CAMERA)
    result = sg.select(graph, near_object_id=0, near_radius_m=25)
    assert 0 not in {n.id for n in result}


def test_select_filters_combine_as_and():
    graph = sg.build(simple_scene(), CAMERA)
    result = sg.select(graph, classes=["person"], side="right", max_distance_m=30)
    assert {n.id for n in result} == {2}


def test_select_by_confidence():
    detections = [
        make_detection(0, "person", 0, 1.0, 10.0, confidence=0.95),
        make_detection(1, "person", 0, 2.0, 10.0, confidence=0.30),
    ]
    graph = sg.build(detections, CAMERA)
    assert {n.id for n in sg.select(graph, min_confidence=0.5)} == {0}


def test_select_near_a_missing_object_returns_nothing():
    graph = sg.build(simple_scene(), CAMERA)
    assert sg.select(graph, near_object_id=999) == []
    # And anchoring on an object with no position is equally unanswerable.
    assert sg.select(graph, near_object_id=4) == []


def test_distance_filters_skip_unlocated_objects():
    graph = sg.build(simple_scene(), CAMERA)
    assert 4 not in {n.id for n in sg.select(graph, max_distance_m=10_000)}


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------


def test_metrics_summarise_the_scene():
    detections = simple_scene()
    metrics = analytics.summarise(detections, [], inference_ms=50.0, total_ms=60.0)
    assert metrics["object_count"] == 5
    assert metrics["located_count"] == 4
    assert metrics["class_count"] == 3          # bus, person, car
    assert metrics["classes"]["person"] == 3
    assert metrics["categories"]["vehicles"] == 2
    assert 0 < metrics["avg_confidence"] <= 1
    assert metrics["nearest"]["id"] == 0
    assert metrics["farthest"]["id"] == 3


def test_footprint_area_uses_the_shoelace_formula():
    # A 100 x 50 m rectangle on the ground.
    square = [
        {"east_m": 0, "north_m": 0}, {"east_m": 100, "north_m": 0},
        {"east_m": 100, "north_m": 50}, {"east_m": 0, "north_m": 50},
    ]
    assert abs(analytics.footprint_area_m2(square) - 5000.0) < 1e-6


def test_open_footprint_has_no_area():
    """A view reaching the horizon is unbounded; reporting a number would lie."""
    assert analytics.footprint_area_m2([{"east_m": 0, "north_m": 0}]) is None
    metrics = analytics.summarise([], [], 10.0, 10.0)
    assert metrics["visible_area_m2"] is None
    assert "unbounded" in metrics["visible_area_note"]


def test_video_fps_uses_wall_clock_not_per_frame_time():
    metrics = analytics.summarise(
        simple_scene(), [], inference_ms=100.0, total_ms=10.0,
        frame_count=40, video_ms=8000.0,
    )
    assert abs(metrics["fps"] - 5.0) < 0.01   # 40 frames in 8 seconds
    assert metrics["frame_count"] == 40


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------


def test_geojson_is_lon_lat_and_closes_its_rings():
    detections = simple_scene()
    footprint = [
        {"lat": 47.0, "lon": 8.0, "east_m": 0, "north_m": 0},
        {"lat": 47.0, "lon": 8.1, "east_m": 10, "north_m": 0},
        {"lat": 47.1, "lon": 8.1, "east_m": 10, "north_m": 10},
    ]
    gj = exports.to_geojson(detections, CAMERA, footprint, "test.jpg")

    assert gj["type"] == "FeatureCollection"
    assert gj["crs"]["properties"]["name"] == "urn:ogc:def:crs:OGC:1.3:CRS84"

    points = [f for f in gj["features"] if f["geometry"]["type"] == "Point"]
    # Four located detections plus the camera. The unlocated one is omitted --
    # it has no position to write.
    assert len(points) == 5

    for feature in points:
        lon, lat = feature["geometry"]["coordinates"]
        assert 8.0 <= lon <= 9.0, "longitude must come first"
        assert 47.0 <= lat <= 48.0

    polys = [f for f in gj["features"] if f["geometry"]["type"] == "Polygon"]
    ring = polys[0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1], "a GeoJSON ring must close"


def test_geojson_marks_positions_as_estimated():
    gj = exports.to_geojson(simple_scene(), CAMERA, [], "test.jpg")
    detections = [f for f in gj["features"]
                  if f["properties"].get("role") is None]
    assert all(f["properties"]["position_type"] == "estimated" for f in detections)


def test_geojson_includes_tracks_as_linestrings():
    trails = [{
        "track_id": 3, "label": "person_03", "class_name": "person",
        "class_id": 0, "distance_m": 12.5,
        "points": [
            {"t": 0.0, "lat": 47.0, "lon": 8.0, "east_m": 0, "north_m": 0},
            {"t": 1.0, "lat": 47.001, "lon": 8.001, "east_m": 5, "north_m": 5},
        ],
    }]
    gj = exports.to_geojson(simple_scene(), CAMERA, [], "clip.mp4", trails)
    lines = [f for f in gj["features"] if f["geometry"]["type"] == "LineString"]
    assert len(lines) == 1
    assert lines[0]["properties"]["track_id"] == 3


def test_csv_has_a_row_per_detection_including_unlocated():
    csv_text = exports.to_csv(simple_scene(), "test.jpg")
    lines = [ln for ln in csv_text.strip().splitlines() if ln]
    assert len(lines) == 6  # header plus five detections

    header = lines[0].split(",")
    assert header[:5] == ["id", "track_id", "class", "category", "confidence"]

    # The unlocated detection keeps its row but leaves position columns empty,
    # rather than filling them with zeros that would read as "at the camera".
    ghost = [ln for ln in lines if ln.startswith("4,")][0]
    assert "no ground fix" in ghost
    fields = ghost.split(",")
    lat_index = header.index("latitude")
    assert fields[lat_index] == ""


def test_json_export_carries_the_honesty_note():
    doc = exports.to_json(simple_scene(), CAMERA, [], "test.jpg")
    assert "ESTIMATED" in doc["position_note"]
    assert doc["camera"] == CAMERA
    assert len(doc["detections"]) == 5
    json.dumps(doc)  # must be serialisable


def test_export_filenames_carry_the_right_extension():
    for fmt, ext in [("geojson", ".geojson"), ("json", ".json"), ("csv", ".csv")]:
        name = exports.filename_for("my photo.jpg", fmt)
        assert name.endswith(ext)
        assert name.startswith("my photo")


# --------------------------------------------------------------------------
# Trails
# --------------------------------------------------------------------------


def test_trails_group_detections_by_track_id():
    from app.video import build_trails

    frames = [
        {"time_s": 0.0, "detections": [
            make_detection(0, "person", 0, 0.0, 10.0, track_id=1),
            make_detection(1, "car", 2, 5.0, 10.0, track_id=2),
        ]},
        {"time_s": 0.2, "detections": [
            make_detection(0, "person", 0, 3.0, 14.0, track_id=1),
        ]},
    ]
    trails = build_trails(frames)
    by_id = {t["track_id"]: t for t in trails}

    assert set(by_id) == {1, 2}
    assert len(by_id[1]["points"]) == 2
    assert by_id[1]["label"] == "person_01"
    # Moved 3 m east and 4 m north, so 5 m by Pythagoras.
    assert abs(by_id[1]["distance_m"] - 5.0) < 0.01


def test_trails_ignore_detections_without_a_track_or_a_position():
    from app.video import build_trails

    frames = [{"time_s": 0.0, "detections": [
        make_detection(0, "person", 0, 0.0, 10.0, track_id=None),
        make_detection(1, "person", 0, 0.0, 10.0, track_id=5, valid=False),
    ]}]
    assert build_trails(frames) == []


# ------------------------------------------------------------------ runner


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc or 'assertion failed'}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {exc!r}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
