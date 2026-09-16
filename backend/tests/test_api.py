"""
End-to-end tests for the HTTP API.

These drive the real FastAPI app with the real model, so they are slower than
the pure-maths tests in test_geo.py -- the first run loads YOLOv8 and the
inference itself takes a second or two on CPU. What they buy is confidence that
the three stages actually fit together: a real JPEG goes in, and geo-referenced
detections come out the other end.

    python backend/tests/test_api.py
    pytest backend/tests/test_api.py

Requires httpx (FastAPI's TestClient dependency):  pip install httpx
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.main import app  # noqa: E402

SAMPLE = config.SAMPLE_DIR / "street-bus.jpg"

STREET_CAMERA = {
    "hfov_deg": 70, "altitude_m": 1.55, "pitch_deg": 6, "heading_deg": 30,
    "lat": 47.3769, "lon": 8.5417, "max_range_m": 300,
}


def _client() -> TestClient:
    # The context manager form runs the lifespan handler, which loads the model.
    return TestClient(app)


def _analyze(client: TestClient, **overrides):
    data = {**STREET_CAMERA, "confidence": 0.35, "iou": 0.45, "max_detections": 100}
    data.update(overrides)
    with open(SAMPLE, "rb") as fh:
        return client.post(
            "/api/analyze",
            files={"file": (SAMPLE.name, fh, "image/jpeg")},
            data=data,
        )


# ---------------------------------------------------------------- system


def test_health_reports_a_loaded_model():
    with _client() as client:
        body = client.get("/api/health").json()
        assert body["model_loaded"] is True
        assert body["status"] == "ok"
        assert body["num_classes"] == 80  # COCO
        assert body["device"] in ("cpu", "cuda")


def test_classes_lists_coco():
    with _client() as client:
        body = client.get("/api/classes").json()
        assert len(body["classes"]) == 80
        assert "person" in body["classes"].values()
        assert "car" in body["ground_classes"]


def test_samples_are_listed_with_presets():
    with _client() as client:
        samples = client.get("/api/samples").json()["samples"]
        assert len(samples) >= 1
        assert all({"filename", "url", "label", "preset"} <= set(s) for s in samples)


# ---------------------------------------------------------------- pipeline


def test_analyze_detects_and_georeferences():
    with _client() as client:
        res = _analyze(client)
        assert res.status_code == 200, res.text
        body = res.json()

        # Stage 2: the detector found the obvious contents of the photo.
        assert len(body["detections"]) > 0
        names = {d["class_name"] for d in body["detections"]}
        assert "bus" in names and "person" in names

        # Stage 3: those detections carry real-world positions.
        assert body["georeferenced_count"] > 0
        for det in body["detections"]:
            assert 0.0 <= det["confidence"] <= 1.0
            assert det["bbox"]["x2"] > det["bbox"]["x1"]
            assert det["bbox"]["y2"] > det["bbox"]["y1"]
            # The ground anchor is the bottom-centre of the box.
            assert det["bbox"]["anchor_y"] == det["bbox"]["y2"]
            if det["world"]["valid"]:
                assert -90 <= det["world"]["lat"] <= 90
                assert -180 <= det["world"]["lon"] <= 180
                assert det["world"]["ground_range_m"] <= STREET_CAMERA["max_range_m"]

        assert body["image_id"]
        assert body["camera"]["vfov_deg"] > 0


def test_street_preset_recovers_realistic_pedestrian_heights():
    """The sanity check that the projection is actually calibrated.

    With the street camera preset, the people in the sample should come out
    roughly person-sized. If a sign flips or the focal length is wrong, this is
    where it shows up as 4 m tall pedestrians.
    """
    with _client() as client:
        body = _analyze(client).json()
        heights = [
            d["world"]["est_height_m"]
            for d in body["detections"]
            if d["class_name"] == "person" and d["world"]["valid"]
        ]
        assert heights, "expected at least one geo-referenced person"
        for h in heights:
            assert 1.4 < h < 2.3, f"implausible person height: {h} m"


def test_reproject_changes_geometry_without_rerunning_the_model():
    with _client() as client:
        first = _analyze(client).json()
        image_id = first["image_id"]

        second = client.post("/api/reproject", json={
            "image_id": image_id,
            "camera": {**STREET_CAMERA, "altitude_m": 80, "pitch_deg": 90, "max_range_m": 500},
        }).json()

        # Same detections, because the detector did not run again.
        assert second["inference_ms"] == 0.0
        assert len(second["detections"]) == len(first["detections"])
        assert [d["class_name"] for d in second["detections"]] == \
               [d["class_name"] for d in first["detections"]]

        # But the world positions moved, because the camera did.
        assert second["detections"][0]["world"]["ground_range_m"] != \
               first["detections"][0]["world"]["ground_range_m"]

        # A nadir view sees a closed quadrilateral of ground.
        assert len(second["footprint"]) == 4


def test_pixel_coordinates_survive_reprojection():
    # Stage 3 must not touch stage 2's output.
    with _client() as client:
        first = _analyze(client).json()
        second = client.post("/api/reproject", json={
            "image_id": first["image_id"],
            "camera": {**STREET_CAMERA, "heading_deg": 200},
        }).json()
        assert [d["bbox"] for d in second["detections"]] == \
               [d["bbox"] for d in first["detections"]]


# ---------------------------------------------------------------- export


def test_geojson_export_is_valid_and_lon_lat_ordered():
    with _client() as client:
        first = _analyze(client).json()
        # Export takes the scene that is on screen, so an export always matches
        # the view -- filters and assistant selections included.
        res = client.post("/api/export/geojson", json={
            "camera": first["camera"],
            "detections": first["detections"],
            "footprint": first["footprint"],
            "source": "street-bus.jpg",
        })
        assert res.status_code == 200
        assert "attachment" in res.headers.get("content-disposition", "")

        gj = res.json()
        assert gj["type"] == "FeatureCollection"
        assert gj["crs"]["properties"]["name"] == "urn:ogc:def:crs:OGC:1.3:CRS84"
        assert len(gj["features"]) >= 2  # detections plus the camera

        for feature in gj["features"]:
            assert feature["type"] == "Feature"
            geom = feature["geometry"]
            if geom["type"] == "Point":
                lon, lat = geom["coordinates"]  # GeoJSON is lon-first
                assert -180 <= lon <= 180 and -90 <= lat <= 90
            elif geom["type"] == "Polygon":
                ring = geom["coordinates"][0]
                assert ring[0] == ring[-1], "polygon ring must close"

        assert any(f["properties"].get("role") == "camera" for f in gj["features"])


# ---------------------------------------------------------------- errors


def test_rejects_non_image_extension():
    with _client() as client:
        res = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
        assert res.status_code == 400
        assert "Unsupported file type" in res.json()["detail"]


def test_rejects_corrupt_image():
    with _client() as client:
        res = client.post("/api/analyze", files={"file": ("x.jpg", b"not a jpeg", "image/jpeg")})
        assert res.status_code == 400


def test_rejects_out_of_range_camera_values():
    with _client() as client:
        res = _analyze(client, pitch_deg=200)
        assert res.status_code == 422


def test_unknown_image_id_is_a_clean_404():
    with _client() as client:
        res = client.post("/api/reproject", json={
            "image_id": "does-not-exist", "camera": STREET_CAMERA,
        })
        assert res.status_code == 404
        assert "expired" in res.json()["detail"].lower()


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
