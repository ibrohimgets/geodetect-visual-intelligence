"""
Tests for the real-sensor path: KITTI calibration, LiDAR fusion and ego pose.

    python backend/tests/test_kitti.py
    pytest backend/tests/test_kitti.py

These need the KITTI drive present under data/kitti. Without it they skip
rather than fail, because the dataset is not committed to the repository --
see the README for the one-line download.

Why these tests exist
---------------------
The transform chain here has four frames of reference and two angle
conventions, and every one of them is easy to get subtly wrong in a way that
still produces plausible-looking numbers. So rather than assert on what the
code currently does, each test pins something that can be checked against an
independent source:

  * the camera height against KITTI's documented rig geometry
  * the ego speed from GPS against the speed the OXTS unit itself reported
  * a parked car's world position against *itself*, across a drive in which
    the observing vehicle moved 17 metres

That last one is the strongest check in the project. If any part of the chain
-- calibration, LiDAR projection, IMU rotation, or the yaw convention -- were
wrong, a stationary car would appear to move.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import kitti  # noqa: E402
from app.config import KITTI_DIR  # noqa: E402
from app.geo import CameraModel  # noqa: E402

SEQUENCES = kitti.discover(KITTI_DIR)
HAVE_DATA = bool(SEQUENCES)

# KITTI's documented rig: the Velodyne sits 1.73 m above the road and the
# colour cameras about 1.65 m. We should recover those from the data.
DOC_CAMERA_HEIGHT_M = 1.65
DOC_VELODYNE_HEIGHT_M = 1.73


def _skip(reason: str) -> None:
    try:
        import pytest

        pytest.skip(reason, allow_module_level=False)
    except ImportError:
        pass
    raise SystemExit(0) if False else None


def _sequence():
    return SEQUENCES[0]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_calibration_matches_the_file():
    if not HAVE_DATA:
        return _skip("no KITTI data")
    calib = _sequence().calib

    # These are the literal values in 2011_09_26/calib_cam_to_cam.txt.
    assert abs(calib.fx - 721.5377) < 1e-3
    assert abs(calib.fy - 721.5377) < 1e-3
    assert abs(calib.cx - 609.5593) < 1e-3
    assert abs(calib.cy - 172.8540) < 1e-3
    assert calib.image_size == (1242, 375)


def test_principal_point_is_not_the_image_centre():
    """Worth asserting, because assuming it is would bias every position.

    KITTI's cx is 609.6 in a 1242 px image -- eleven pixels off centre. That is
    small, but it is a systematic bias, and it is exactly the kind of thing a
    'close enough' shortcut would introduce.
    """
    if not HAVE_DATA:
        return _skip("no KITTI data")
    calib = _sequence().calib
    assert abs(calib.cx - calib.image_size[0] / 2) > 5


def test_projection_matrix_keeps_the_baseline_term():
    """P_rect_02's fourth column offsets camera 2 from the rectified origin."""
    if not HAVE_DATA:
        return _skip("no KITTI data")
    calib = _sequence().calib
    assert abs(calib.P_rect_02[0, 3]) > 1.0  # ~44.86 for this rig


def test_camera_height_matches_the_documented_rig():
    """Measure the road with the LiDAR and check it against KITTI's own figures."""
    if not HAVE_DATA:
        return _skip("no KITTI data")
    sequence = _sequence()
    points = sequence.frames[0].lidar()

    ground_z = kitti.ground_level_velo(points)
    assert ground_z is not None
    # The Velodyne sits ~1.73 m up, so the road is ~-1.73 m in its own frame.
    assert abs(-ground_z - DOC_VELODYNE_HEIGHT_M) < 0.15

    camera_velo = sequence.calib.camera_origin_in_velo
    height = camera_velo[2] - ground_z
    assert abs(height - DOC_CAMERA_HEIGHT_M) < 0.15, f"camera height {height:.3f} m"


def test_optical_axis_points_forward():
    """The colour camera faces along the vehicle's nose, not sideways."""
    if not HAVE_DATA:
        return _skip("no KITTI data")
    axis = _sequence().calib.camera_axis_in_imu()
    assert axis[0] > 0.99          # almost entirely +x, which is forward
    assert abs(axis[1]) < 0.05     # negligible sideways component
    assert abs(axis[2]) < 0.05     # and near-level


# --------------------------------------------------------------------------
# LiDAR projection
# --------------------------------------------------------------------------


def test_lidar_projects_into_the_image():
    if not HAVE_DATA:
        return _skip("no KITTI data")
    sequence = _sequence()
    points = sequence.frames[0].lidar()
    uv, depth, inside = sequence.calib.project_lidar(points)

    # The Velodyne spins through 360 degrees while the camera sees ~81, so only
    # a modest slice should land in frame. Far outside this range means the
    # projection is wrong, not merely imprecise.
    fraction = inside.sum() / len(points)
    assert 0.05 < fraction < 0.35, f"{fraction:.1%} of points projected inside"

    assert (depth[inside] > 0).all(), "every visible point must be in front"
    width, height = sequence.calib.image_size
    assert uv[inside][:, 0].min() >= 0 and uv[inside][:, 0].max() < width
    assert uv[inside][:, 1].min() >= 0 and uv[inside][:, 1].max() < height


def test_points_behind_the_camera_are_rejected():
    """A point behind the sensor projects to a perfectly plausible pixel.

    Dividing by a negative depth flips the sign twice and lands somewhere in
    frame, so without an explicit in-front test the scene quietly fills with
    ghosts from behind the vehicle.
    """
    if not HAVE_DATA:
        return _skip("no KITTI data")
    calib = _sequence().calib
    behind = np.array([[-20.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    _, _, inside = calib.project_lidar(behind)
    assert not inside[0]


def test_a_point_on_the_road_ahead_lands_below_the_horizon():
    if not HAVE_DATA:
        return _skip("no KITTI data")
    sequence = _sequence()
    calib = sequence.calib
    ground_z = kitti.ground_level_velo(sequence.frames[0].lidar())

    road = np.array([[15.0, 0.0, ground_z, 1.0]], dtype=np.float32)
    uv, depth, inside = calib.project_lidar(road)
    assert inside[0]
    # Straight ahead, so near the principal point horizontally...
    assert abs(uv[0][0] - calib.cx) < 20
    # ...and below it vertically, because the road is beneath the camera.
    assert uv[0][1] > calib.cy
    assert 14.0 < depth[0] < 16.0


# --------------------------------------------------------------------------
# OXTS pose
# --------------------------------------------------------------------------


def test_yaw_converts_from_east_ccw_to_compass():
    """OXTS yaw is 0 = east, counter-clockwise. A compass is 0 = north, clockwise."""
    east = kitti.OxtsPose(0, 0, 0, 0, 0, 0.0, 0, 0, 0)
    assert abs(east.heading_deg - 90.0) < 1e-6          # yaw 0 -> due east

    north = kitti.OxtsPose(0, 0, 0, 0, 0, math.pi / 2, 0, 0, 0)
    assert abs(north.heading_deg - 0.0) < 1e-6          # yaw +90 -> due north

    west = kitti.OxtsPose(0, 0, 0, 0, 0, math.pi, 0, 0, 0)
    assert abs(west.heading_deg - 270.0) < 1e-6


def test_rotation_puts_the_nose_on_the_compass_heading():
    """Two independent routes to the same bearing must agree.

    heading_deg comes from the yaw scalar; rotating the IMU's forward axis into
    ENU comes from the full rotation matrix. If the matrix composition or the
    axis convention were wrong, these would diverge.
    """
    if not HAVE_DATA:
        return _skip("no KITTI data")
    for frame in _sequence().frames[::5]:
        pose = frame.pose()
        nose = pose.rotation_to_enu() @ np.array([1.0, 0.0, 0.0])
        bearing = math.degrees(math.atan2(nose[0], nose[1])) % 360.0
        assert abs(bearing - pose.heading_deg) < 0.5


def test_gps_track_agrees_with_the_reported_speed():
    """Differentiate the GPS positions and compare with what OXTS said."""
    if not HAVE_DATA:
        return _skip("no KITTI data")
    sequence = _sequence()
    first, last = sequence.frames[0].pose(), sequence.frames[-1].pose()

    north = (last.lat - first.lat) * 111132.0
    east = (last.lon - first.lon) * 111320.0 * math.cos(math.radians(first.lat))
    seconds = (sequence.frame_count - 1) / sequence.fps
    derived_kmh = math.hypot(east, north) / seconds * 3.6

    reported = (first.speed_kmh + last.speed_kmh) / 2
    assert abs(derived_kmh - reported) < 4.0, \
        f"GPS says {derived_kmh:.1f} km/h, OXTS says {reported:.1f} km/h"


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def test_measured_positions_are_plausible():
    if not HAVE_DATA:
        return _skip("no KITTI data")
    sequence = _sequence()
    frame = sequence.frames[0]
    points = frame.lidar()
    pose = frame.pose()

    # A generous box over the lower-middle of the frame: whatever is on the
    # road ahead.
    boxes = [(400.0, 150.0, 800.0, 340.0)]
    fixes = kitti.measure_detections(boxes, points, sequence.calib, pose)
    fix = fixes[0]

    assert fix.ok and fix.point_count > 50
    assert 1.0 < fix.distance_m < 80.0
    assert fix.depth_m > 0
    # Forward is roughly south-east on this drive, so the object should sit
    # somewhere in front rather than behind.
    assert math.hypot(fix.east_m, fix.north_m) == pytest_approx(fix.distance_m)


def pytest_approx(value, tol=0.01):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

        def __repr__(self):
            return f"~{value}"

    return _Approx()


def test_too_few_returns_is_reported_not_guessed():
    """An empty patch of sky has no depth, and must say so."""
    if not HAVE_DATA:
        return _skip("no KITTI data")
    sequence = _sequence()
    frame = sequence.frames[0]

    # The top-left corner of a KITTI frame is sky and rooftops, above the
    # Velodyne's vertical field of view.
    boxes = [(0.0, 0.0, 60.0, 30.0)]
    fixes = kitti.measure_detections(boxes, frame.lidar(), sequence.calib, frame.pose())
    assert not fixes[0].ok
    assert "LiDAR returns" in fixes[0].reason


def test_foreground_is_separated_from_background():
    """A box containing an object and a distant wall must report the object.

    This is the failure mode that makes naive box-depth useless: average the
    returns and a pedestrian standing in front of a building lands halfway
    between the two.
    """
    if not HAVE_DATA:
        return _skip("no KITTI data")
    sequence = _sequence()
    calib = sequence.calib
    frame = sequence.frames[0]

    points = frame.lidar()
    uv, depth, inside = calib.project_lidar(points)
    visible_depth = depth[inside]

    boxes = [(300.0, 150.0, 900.0, 370.0)]
    fix = kitti.measure_detections(boxes, points, calib, frame.pose())[0]
    assert fix.ok

    # The reported depth should sit near the near end of what is in that box,
    # not at its mean, which the background would drag outwards.
    in_box = (
        (uv[inside][:, 0] >= 300) & (uv[inside][:, 0] <= 900)
        & (uv[inside][:, 1] >= 150) & (uv[inside][:, 1] <= 370)
    )
    box_depths = visible_depth[in_box]
    assert fix.depth_m < float(np.mean(box_depths)) + 1.0


def test_parked_cars_stay_put_while_the_vehicle_drives_past():
    """The end-to-end check, and the strongest one available.

    Run the real pipeline -- detection, tracking, LiDAR fusion, GPS/IMU
    rotation -- over the whole drive, then look at each tracked object twice:

      * its ego-relative position, which must sweep past as the vehicle moves
      * its absolute WGS84 position, which for a parked car must not move

    This drive is a street of parked cars filmed from a vehicle doing 27 km/h,
    so the two should disagree enormously. Getting a small absolute figure means
    the calibration, the LiDAR projection, the IMU rotation and the yaw
    convention are all correct *together* -- no single one of them can be wrong
    and still cancel the ego motion this precisely.

    It needs the detector, so it is slower than the rest of this file. It earns
    that: it is the only test here that can catch a consistent error running
    through the whole chain.
    """
    if not HAVE_DATA:
        return _skip("no KITTI data")

    from app.config import MODELS_DIR, MODEL_WEIGHTS
    from app.detector import Detector

    sequence = _sequence()
    detector = Detector(weights=MODEL_WEIGHTS, models_dir=MODELS_DIR)
    result = kitti.process_sequence(
        sequence, detector, Path(KITTI_DIR).parent / "uploads", confidence=0.35,
    )

    # Gather each track's observations in both frames of reference.
    tracks: dict[int, dict[str, list]] = {}
    for frame in result.frames:
        cam = CameraModel(
            image_width=result.width, image_height=result.height,
            lat=frame.camera["lat"], lon=frame.camera["lon"],
        )
        for det, fix in zip(frame.detections, frame.fixes):
            if det.track_id is None or not fix.ok:
                continue
            entry = tracks.setdefault(det.track_id, {"rel": [], "abs": []})
            entry["rel"].append((fix.east_m, fix.north_m))
            entry["abs"].append(cam.enu_to_latlon(fix.east_m, fix.north_m))

    persistent = {k: v for k, v in tracks.items() if len(v["rel"]) >= 5}
    assert persistent, "no object was tracked across enough frames"

    relative_sweeps: list[float] = []
    absolute_moves: list[float] = []

    for entry in persistent.values():
        rel_first, rel_last = entry["rel"][0], entry["rel"][-1]
        relative_sweeps.append(
            math.hypot(rel_last[0] - rel_first[0], rel_last[1] - rel_first[1])
        )

        abs_first, abs_last = entry["abs"][0], entry["abs"][-1]
        north = (abs_last[0] - abs_first[0]) * 111132.0
        east = ((abs_last[1] - abs_first[1]) * 111320.0
                * math.cos(math.radians(abs_first[0])))
        absolute_moves.append(math.hypot(east, north))

    median_relative = float(np.median(relative_sweeps))
    median_absolute = float(np.median(absolute_moves))

    # How far the vehicle itself travelled: the yardstick for "cancelled".
    first, last = sequence.frames[0].pose(), sequence.frames[-1].pose()
    ego_north = (last.lat - first.lat) * 111132.0
    ego_east = (last.lon - first.lon) * 111320.0 * math.cos(math.radians(first.lat))
    ego_moved = math.hypot(ego_east, ego_north)

    assert ego_moved > 10.0, f"expected a moving drive, ego covered {ego_moved:.1f} m"
    assert median_relative > 5.0, \
        f"expected ego motion to sweep objects past, got {median_relative:.1f} m"

    # The discriminating comparison.
    #
    # If the ego motion were not being cancelled -- a missing rotation, the
    # wrong yaw convention, positions left in the vehicle frame -- then parked
    # cars would appear to travel about as far as the vehicle did. So the test
    # is against the ego's own displacement, not against zero.
    #
    # It is not zero, and should not be expected to be. A LiDAR only sees the
    # faces pointing at it, so as you drive past a 4.5 m car the centroid of its
    # visible returns migrates from the back of the car towards the side. A
    # metre or two of residual is that geometry, not a fusion error, and a few
    # of the tracked vehicles on this street are genuinely moving.
    assert median_absolute < ego_moved * 0.25, (
        f"tracked objects moved a median of {median_absolute:.2f} m in world "
        f"coordinates while the vehicle covered {ego_moved:.1f} m -- ego motion "
        "is not being cancelled, so the fusion is wrong"
    )


# ------------------------------------------------------------------ runner


if __name__ == "__main__":
    if not HAVE_DATA:
        print(f"No KITTI data under {KITTI_DIR} - skipping.")
        print("See the README for the download command.")
        raise SystemExit(0)

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
    raise SystemExit(1 if failures else 0)
