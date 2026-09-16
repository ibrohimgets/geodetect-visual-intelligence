"""
Tests for the geo-referencing maths.

Run them with either:

    python backend/tests/test_geo.py      # no pytest needed
    pytest backend/tests/test_geo.py

The projection is the part of this project that can be silently wrong -- a sign
flip in the heading or a swapped axis still produces plausible-looking numbers
on a map. So rather than checking that the code does what it currently does,
every test here pins a value that can be worked out by hand, and the last one
does a full round trip: take an object of known size at a known place, project
it into pixels, and check the pipeline recovers what we started with.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Allow running this file directly, not just under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.geo import (  # noqa: E402
    MAX_PLAUSIBLE_HEIGHT_M,
    CameraModel,
    ground_footprint,
    project_detection,
)


def square_cam(**kwargs) -> CameraModel:
    """A 1000x1000 camera with a 90 degree FOV -- easy numbers to reason about."""
    base = dict(image_width=1000, image_height=1000, hfov_deg=90.0, altitude_m=10.0)
    base.update(kwargs)
    return CameraModel(**base)


# ---------------------------------------------------------------- intrinsics


def test_focal_length_from_fov():
    # With a 90 degree horizontal FOV, half the FOV is 45 degrees and tan(45) = 1,
    # so the focal length equals half the image width. Compared with a tolerance
    # because tan(radians(45)) is 0.9999999999999999, not exactly 1.
    cam = square_cam()
    assert abs(cam.fx - 500.0) < 1e-9
    assert cam.fy == cam.fx


def test_vertical_fov_follows_aspect_ratio():
    cam = CameraModel(image_width=1000, image_height=500, hfov_deg=90.0)
    expected = math.degrees(2 * math.atan(250 / 500))
    assert abs(cam.vfov_deg - expected) < 1e-9
    # A wider-than-tall sensor must see less vertically than horizontally.
    assert cam.vfov_deg < cam.hfov_deg


def test_principal_point_is_image_centre():
    cam = CameraModel(image_width=640, image_height=480)
    assert (cam.cx, cam.cy) == (320.0, 240.0)


# ---------------------------------------------------------------- projection


def test_nadir_centre_lands_under_the_camera():
    cam = square_cam(pitch_deg=90.0, altitude_m=100.0)
    hit = cam.ray_ground_hit(500, 500)
    assert hit is not None
    assert abs(hit[0]) < 1e-9
    assert abs(hit[1]) < 1e-9


def test_level_camera_centre_is_the_horizon():
    # Pointing at the horizon, the central ray is parallel to the ground and
    # never reaches it. None is the correct answer, not a huge number.
    cam = square_cam(pitch_deg=0.0)
    assert cam.ray_ground_hit(500, 500) is None


def test_pitch_45_puts_the_target_one_altitude_away():
    # At 45 degrees the ground range equals the altitude: tan(45) = 1.
    cam = square_cam(pitch_deg=45.0, altitude_m=10.0)
    hit = cam.ray_ground_hit(500, 500)
    assert abs(hit[0] - 0.0) < 1e-6
    assert abs(hit[1] - 10.0) < 1e-6
    assert abs(cam.ground_range(hit) - 10.0) < 1e-6
    assert abs(cam.slant_range(hit) - math.hypot(10, 10)) < 1e-6


def test_pitch_30_matches_altitude_over_tan():
    cam = square_cam(pitch_deg=30.0, altitude_m=10.0)
    hit = cam.ray_ground_hit(500, 500)
    assert abs(hit[1] - 10.0 / math.tan(math.radians(30))) < 1e-6


def test_heading_rotates_the_target_clockwise_from_north():
    """0 = North, 90 = East, 180 = South, 270 = West."""
    expected = {
        0: (0.0, 10.0),
        90: (10.0, 0.0),
        180: (0.0, -10.0),
        270: (-10.0, 0.0),
    }
    for heading, (east, north) in expected.items():
        cam = square_cam(pitch_deg=45.0, heading_deg=heading)
        hit = cam.ray_ground_hit(500, 500)
        assert abs(hit[0] - east) < 1e-6, f"east wrong at heading {heading}"
        assert abs(hit[1] - north) < 1e-6, f"north wrong at heading {heading}"
        assert abs(cam.bearing_from_camera(hit) - heading) < 1e-6


def test_roll_does_not_move_the_centre_pixel():
    # The optical axis is the roll axis, so spinning about it leaves the centre
    # of the image pointing at exactly the same spot.
    straight = square_cam(pitch_deg=45.0).ray_ground_hit(500, 500)
    rolled = square_cam(pitch_deg=45.0, roll_deg=25.0).ray_ground_hit(500, 500)
    assert np.allclose(straight, rolled, atol=1e-9)


def test_roll_swaps_which_way_the_frame_leans():
    # Off-centre pixels must move when the camera rolls, otherwise roll is
    # being silently ignored.
    straight = square_cam(pitch_deg=60.0).ray_ground_hit(900, 500)
    rolled = square_cam(pitch_deg=60.0, roll_deg=20.0).ray_ground_hit(900, 500)
    assert not np.allclose(straight, rolled, atol=1e-3)


def test_range_limit_rejects_distant_hits():
    # A shallow ray does reach the ground, but so far away that the flat-earth
    # assumption is meaningless. max_range_m must reject it.
    shallow = square_cam(pitch_deg=1.0, altitude_m=10.0, max_range_m=100.0)
    assert shallow.ray_ground_hit(500, 500) is None

    generous = square_cam(pitch_deg=1.0, altitude_m=10.0, max_range_m=10000.0)
    assert generous.ray_ground_hit(500, 500) is not None


# ---------------------------------------------------------------- geodesy


def test_metres_north_convert_to_plausible_latitude():
    cam = CameraModel(image_width=100, image_height=100, lat=47.3769, lon=8.5417)
    lat, lon = cam.enu_to_latlon(0.0, 1000.0)
    # One degree of latitude is about 111.1 km, so 1 km is about 0.009 degrees.
    assert abs((lat - cam.lat) - 1000.0 / 111132.0) < 2e-4
    assert lon == cam.lon  # moving north must not change longitude


def test_metres_east_convert_to_plausible_longitude():
    cam = CameraModel(image_width=100, image_height=100, lat=47.3769, lon=8.5417)
    lat, lon = cam.enu_to_latlon(1000.0, 0.0)
    expected = 1000.0 / (111320.0 * math.cos(math.radians(47.3769)))
    assert abs((lon - cam.lon) - expected) < 1e-4
    assert abs(lat - cam.lat) < 1e-12


def test_longitude_degrees_shrink_towards_the_poles():
    # The same eastward distance is worth more degrees the further north you go.
    equator = CameraModel(image_width=100, image_height=100, lat=0.0, lon=0.0)
    arctic = CameraModel(image_width=100, image_height=100, lat=70.0, lon=0.0)
    d_equator = equator.enu_to_latlon(1000.0, 0.0)[1]
    d_arctic = arctic.enu_to_latlon(1000.0, 0.0)[1]
    assert d_arctic > d_equator


# ---------------------------------------------------------------- footprint


def test_nadir_footprint_has_four_corners():
    cam = square_cam(pitch_deg=90.0, altitude_m=80.0, hfov_deg=60.0)
    assert len(ground_footprint(cam)) == 4


def test_level_footprint_drops_the_sky_corners():
    # Looking at the horizon, the top half of the frame is sky. Only the two
    # bottom corners reach the ground.
    cam = square_cam(pitch_deg=0.0, altitude_m=2.0, hfov_deg=60.0, max_range_m=500.0)
    assert len(ground_footprint(cam)) == 2


# ---------------------------------------------------------------- detections


def test_detection_above_the_horizon_is_flagged_not_guessed():
    cam = square_cam(pitch_deg=0.0, altitude_m=2.0)
    fix = project_detection(cam, 400, 100, 600, 200)  # high in frame = sky
    assert fix.valid is False
    assert fix.reason  # and it says why


def test_height_estimate_is_clamped_to_something_possible():
    """A mismatched camera pose must not produce a 400 m tall pedestrian.

    Reprojecting a street-level frame as though it came from a drone at 300 m is
    a nonsense premise, and the geometry will faithfully return a nonsense
    height. The clamp is what keeps that from wrecking the 3D scene.
    """
    cam = square_cam(pitch_deg=90.0, altitude_m=300.0, max_range_m=5000.0)
    fix = project_detection(cam, 100, 100, 900, 900)
    assert fix.valid
    assert fix.height_m <= MAX_PLAUSIBLE_HEIGHT_M


def test_round_trip_recovers_a_known_object():
    """The real test: synthesise an object, project it, and invert it.

    We place a 1.8 m x 0.6 m person 20 m due north of a street camera, work out
    which pixels they would occupy using the forward projection, then hand that
    bounding box to the pipeline. If the maths is consistent, we get 1.8, 0.6
    and 20 back out. This catches sign flips and axis swaps that spot checks
    can miss.
    """
    cam = CameraModel(
        image_width=1920, image_height=1080, hfov_deg=70.0,
        altitude_m=2.5, pitch_deg=0.0, heading_deg=0.0,
        lat=47.3769, lon=8.5417,
    )
    rotation = cam.rotation_matrix()

    def world_to_pixel(point):
        """Forward projection -- the inverse of what geo.py does."""
        d_cam = rotation.T @ (point - cam.position())
        return (
            cam.fx * d_cam[0] / d_cam[2] + cam.cx,
            cam.fy * d_cam[1] / d_cam[2] + cam.cy,
        )

    true_h, true_w, distance = 1.8, 0.6, 20.0
    _, v_bottom = world_to_pixel(np.array([0.0, distance, 0.0]))
    _, v_top = world_to_pixel(np.array([0.0, distance, true_h]))
    u_left, _ = world_to_pixel(np.array([-true_w / 2, distance, 0.0]))
    u_right, _ = world_to_pixel(np.array([true_w / 2, distance, 0.0]))

    fix = project_detection(cam, u_left, v_top, u_right, v_bottom)

    assert fix.valid
    assert abs(fix.north_m - distance) < 0.01
    assert abs(fix.east_m) < 0.01
    assert abs(fix.height_m - true_h) < 0.01
    assert abs(fix.width_m - true_w) < 0.01
    # And the lat/lon should be about 20 m north of the camera.
    assert fix.lat > cam.lat
    assert abs(fix.lon - cam.lon) < 1e-6


def test_round_trip_holds_at_an_angle():
    """Same idea, but with the camera pitched down and pointing south-east."""
    cam = CameraModel(
        image_width=1600, image_height=900, hfov_deg=84.0,
        altitude_m=40.0, pitch_deg=35.0, heading_deg=135.0,
        lat=51.5, lon=-0.12,
    )
    rotation = cam.rotation_matrix()

    def world_to_pixel(point):
        d_cam = rotation.T @ (point - cam.position())
        return (
            cam.fx * d_cam[0] / d_cam[2] + cam.cx,
            cam.fy * d_cam[1] / d_cam[2] + cam.cy,
        )

    # A 4.5 m long, 3.2 m tall lorry, 50 m out on the camera bearing.
    bearing = math.radians(135.0)
    east, north = 50 * math.sin(bearing), 50 * math.cos(bearing)
    true_h, true_w = 3.2, 4.5

    # The width is measured across the line of sight, not along the axes.
    across = np.array([math.cos(bearing), -math.sin(bearing), 0.0])
    base = np.array([east, north, 0.0])

    _, v_bottom = world_to_pixel(base)
    _, v_top = world_to_pixel(base + np.array([0.0, 0.0, true_h]))
    u_left, _ = world_to_pixel(base - across * true_w / 2)
    u_right, _ = world_to_pixel(base + across * true_w / 2)

    fix = project_detection(cam, min(u_left, u_right), v_top, max(u_left, u_right), v_bottom)

    assert fix.valid
    assert abs(fix.east_m - east) < 0.05
    assert abs(fix.north_m - north) < 0.05
    assert abs(fix.ground_range_m - 50.0) < 0.05
    assert abs(fix.bearing_deg - 135.0) < 0.05
    assert abs(fix.height_m - true_h) < 0.05
    assert abs(fix.width_m - true_w) < 0.05


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
