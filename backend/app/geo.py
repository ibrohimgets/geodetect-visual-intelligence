"""
Geo-referencing: turn image pixels into real-world coordinates.

The whole idea in one paragraph
-------------------------------
A camera turns 3D rays into 2D pixels, and that throws away depth. You cannot
recover depth from a single image -- unless you add an assumption. The one we
use here is the *flat ground plane*: every object we detect is standing on the
ground at elevation 0. That is enough to make the problem solvable, because now
each pixel maps to exactly one place: shoot a ray out of the camera through that
pixel and see where it hits the ground.

So the pipeline for one detection is:

    bbox bottom-centre pixel      (where the object touches the ground)
        -> ray in camera coords   (undo the lens / intrinsics)
        -> ray in world coords    (apply pitch + heading / extrinsics)
        -> intersect ground z=0   (the flat-earth assumption)
        -> local ENU metres       (East / North offset from the camera)
        -> latitude / longitude   (local tangent plane around the camera)

This is the classic monocular ground-plane projection, also called inverse
perspective mapping. It is what you would reach for with a nadir or oblique
drone frame, or a fixed street camera, when you have no depth sensor.

Coordinate frames used here
---------------------------
World (ENU), right-handed, origin on the ground directly below the camera:
    X = East (metres), Y = North (metres), Z = Up (metres)

Camera, the usual computer-vision convention:
    x = right across the image, y = down the image, z = forward along the lens

Accuracy note
-------------
Results are only as good as the assumptions. Flat terrain, no lens distortion,
and the camera pose you type in are all taken at face value. Objects near the
horizon carry very large errors, which is why max_range_m exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# WGS84 ellipsoid constants, used for the local tangent plane conversion.
_WGS84_A = 6378137.0                      # semi-major axis, metres
_WGS84_F = 1.0 / 298.257223563            # flattening
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)   # first eccentricity squared

# Sanity rail on the estimated object height, in metres.
#
# Monocular height estimates fall apart when the camera pose you supply does not
# match the pose the photo was actually taken from: tell the app a street-level
# snapshot came from a drone at 80 m and the maths will faithfully report a 44 m
# tall pedestrian. The geometry is not wrong, the premise is. This ceiling stops
# such a result from wrecking the 3D scene. It sits well above the tallest thing
# in the COCO class list, so it never truncates a plausible answer.
MAX_PLAUSIBLE_HEIGHT_M = 50.0


@dataclass
class CameraModel:
    """Everything needed to shoot a ray through a pixel and land it on the ground.

    Intrinsics (what the lens and sensor do):
        image_width, image_height : pixels
        hfov_deg                  : horizontal field of view in degrees

    Extrinsics (where the camera is and where it points):
        altitude_m  : height of the camera above the ground plane
        pitch_deg   : how far the lens is tilted DOWN from horizontal.
                      0  = looking at the horizon (street camera)
                      90 = looking straight down (nadir / map-style drone shot)
        heading_deg : compass bearing the lens points along.
                      0 = North, 90 = East, 180 = South, 270 = West
        roll_deg    : rotation about the lens axis, usually ~0 on a gimbal

    Origin (where on Earth the camera is):
        lat, lon    : WGS84 degrees

    Limits:
        max_range_m : ignore ground hits further out than this. Rays passing
                      close to the horizon hit the ground kilometres away and
                      the answer stops meaning anything, so we cut them off.
    """

    image_width: int
    image_height: int
    hfov_deg: float = 78.0
    altitude_m: float = 60.0
    pitch_deg: float = 45.0
    heading_deg: float = 0.0
    roll_deg: float = 0.0
    lat: float = 47.3769
    lon: float = 8.5417
    max_range_m: float = 500.0

    # Calibrated intrinsics, when the camera came with a calibration file rather
    # than a typed-in field of view. KITTI's P_rect matrix supplies all four, and
    # its principal point is genuinely off-centre (609.6 in a 1242 px image), so
    # assuming the centre would bias every position sideways. Left as None, the
    # properties below fall back to deriving everything from hfov_deg.
    fx_px: Optional[float] = None
    fy_px: Optional[float] = None
    cx_px: Optional[float] = None
    cy_px: Optional[float] = None

    # ---------------------------------------------------------------- intrinsics

    @property
    def fx(self) -> float:
        """Focal length in pixels, horizontal.

        Taken from the calibration when there is one. Otherwise derived from the
        field of view: half the sensor width is (image_width / 2) pixels and it
        subtends half the FOV, so tan(hfov / 2) = (width / 2) / fx.
        """
        if self.fx_px is not None:
            return self.fx_px
        half = math.radians(self.hfov_deg) / 2.0
        return (self.image_width / 2.0) / math.tan(half)

    @property
    def fy(self) -> float:
        """Focal length in pixels, vertical. Square pixels assumed unless calibrated."""
        return self.fy_px if self.fy_px is not None else self.fx

    @property
    def vfov_deg(self) -> float:
        """Vertical field of view, implied by the image height and focal length."""
        return math.degrees(2.0 * math.atan((self.image_height / 2.0) / self.fy))

    @property
    def cx(self) -> float:
        """Principal point x -- the image centre unless calibration says otherwise."""
        return self.cx_px if self.cx_px is not None else self.image_width / 2.0

    @property
    def cy(self) -> float:
        """Principal point y -- the image centre unless calibration says otherwise."""
        return self.cy_px if self.cy_px is not None else self.image_height / 2.0

    @property
    def is_calibrated(self) -> bool:
        """True when the intrinsics came from a calibration file."""
        return self.fx_px is not None

    # ---------------------------------------------------------------- extrinsics

    def rotation_matrix(self) -> np.ndarray:
        """Build the 3x3 matrix that rotates a camera-frame vector into world ENU.

        Composed from three pieces:

        1. base puts a level, North-facing camera into the world. Its forward
           axis maps to North, its right axis to East, its down axis to Down.
        2. r_roll spins the camera about its own lens axis (applied innermost,
           in camera coordinates).
        3. r_pitch tips the lens downwards, rotating about the East axis.
        4. r_head swings the camera round the vertical axis to the compass
           bearing. Compass bearings run clockwise (N -> E) while a positive
           maths rotation about +Z (Up) runs counter-clockwise, hence the minus.
        """
        # Columns are where the camera x, y, z axes land in world ENU:
        #   camera +x (right)   -> (1, 0, 0)   East
        #   camera +y (down)    -> (0, 0, -1)  Down
        #   camera +z (forward) -> (0, 1, 0)   North
        base = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ])

        # Tilting DOWN is a negative rotation about the East (+X) axis.
        p = math.radians(-self.pitch_deg)
        r_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(p), -math.sin(p)],
            [0.0, math.sin(p), math.cos(p)],
        ])

        # Compass heading is clockwise; rotation about Up is counter-clockwise.
        h = math.radians(-self.heading_deg)
        r_head = np.array([
            [math.cos(h), -math.sin(h), 0.0],
            [math.sin(h), math.cos(h), 0.0],
            [0.0, 0.0, 1.0],
        ])

        # Roll is about the camera forward axis, so it acts in camera coords.
        r = math.radians(self.roll_deg)
        r_roll = np.array([
            [math.cos(r), -math.sin(r), 0.0],
            [math.sin(r), math.cos(r), 0.0],
            [0.0, 0.0, 1.0],
        ])

        return r_head @ r_pitch @ base @ r_roll

    def position(self) -> np.ndarray:
        """Camera position in world ENU. The origin is the ground point below it."""
        return np.array([0.0, 0.0, float(self.altitude_m)])

    # ---------------------------------------------------------------- projection

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Turn a pixel into a unit direction vector in world ENU.

        Undo the intrinsics to get a direction in camera coordinates, then rotate
        it into the world. The result is a direction only -- it carries no
        length, because one pixel tells us nothing about distance.
        """
        d_cam = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        d_world = self.rotation_matrix() @ d_cam
        return d_world / np.linalg.norm(d_world)

    def ray_ground_hit(self, u: float, v: float) -> Optional[np.ndarray]:
        """Where does the ray through pixel (u, v) hit the ground plane z = 0?

        Returns the ENU point as [East, North, 0], or None when the ray never
        gets there. That happens in two cases, and both are real situations
        rather than bugs:
          * the ray points at or above the horizon (sky, upper storeys)
          * the ray lands so far out that the flat-ground assumption is useless
        """
        origin = self.position()
        d = self.pixel_to_ray(u, v)

        # The ray must head downwards to ever reach z = 0. The epsilon also
        # rejects rays running almost parallel to the ground, which would
        # otherwise produce an astronomically large t.
        if d[2] > -1e-6:
            return None

        t = origin[2] / (-d[2])
        hit = origin + t * d

        if math.hypot(hit[0], hit[1]) > self.max_range_m:
            return None

        return np.array([hit[0], hit[1], 0.0])

    def ground_range(self, point_enu: np.ndarray) -> float:
        """Horizontal distance from the camera to a ground point, in metres."""
        return float(math.hypot(point_enu[0], point_enu[1]))

    def slant_range(self, point_enu: np.ndarray) -> float:
        """Straight-line distance from the lens to a ground point, in metres."""
        return float(np.linalg.norm(point_enu - self.position()))

    # ---------------------------------------------------------------- geodesy

    def enu_to_latlon(self, east_m: float, north_m: float) -> tuple[float, float]:
        """Convert a local East/North offset into WGS84 latitude and longitude.

        A local tangent plane ("flat earth") conversion. We work out how many
        metres one degree of latitude and one degree of longitude are worth *at
        this latitude*, using the WGS84 radii of curvature, then divide. Good to
        well under a metre across the few hundred metres this tool covers, and
        it avoids pulling in a full projection library.
        """
        lat_rad = math.radians(self.lat)
        sin_lat = math.sin(lat_rad)

        # Radii of curvature in the meridian (north-south) and the prime
        # vertical (east-west). They differ because the Earth is an ellipsoid.
        denom = math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        m_per_deg_lat = math.radians(1.0) * (_WGS84_A * (1.0 - _WGS84_E2) / denom ** 3)
        m_per_deg_lon = math.radians(1.0) * (_WGS84_A / denom) * math.cos(lat_rad)

        lat = self.lat + north_m / m_per_deg_lat
        # Near the poles a degree of longitude collapses to nothing; guard it.
        lon = self.lon + (east_m / m_per_deg_lon if abs(m_per_deg_lon) > 1e-9 else 0.0)
        return lat, lon

    def bearing_from_camera(self, point_enu: np.ndarray) -> float:
        """Compass bearing from the camera to a ground point, 0-360 degrees."""
        # atan2(East, North) gives the clockwise-from-North convention directly.
        deg = math.degrees(math.atan2(point_enu[0], point_enu[1]))
        return deg % 360.0


@dataclass
class GroundFix:
    """The geo-referenced result for a single detection.

    valid is False when the object could not be placed on the ground -- it sits
    above the horizon, or past max_range. Those detections still exist and are
    still drawn on the image, they simply have no map or 3D position.
    """

    valid: bool
    east_m: float = 0.0
    north_m: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    ground_range_m: float = 0.0
    slant_range_m: float = 0.0
    bearing_deg: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0
    reason: str = ""


def project_detection(
    cam: CameraModel,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> GroundFix:
    """Geo-reference one bounding box.

    The bottom edge of a box is where the object meets the ground, so the
    bottom-centre pixel is our ground contact point. From there we also estimate
    real-world size, which is what gives the 3D scene correctly proportioned
    boxes instead of uniform blocks:

      * width comes from projecting the bottom-left and bottom-right corners
        onto the ground and measuring the gap between them.
      * height comes from the top edge. We already know the object ground
        position, so we walk the top-edge ray out to that same horizontal
        distance and read off how high it has climbed.
    """
    base_u = (x1 + x2) / 2.0
    base_v = y2  # bottom edge of the box

    hit = cam.ray_ground_hit(base_u, base_v)
    if hit is None:
        return GroundFix(
            valid=False,
            reason="Ray misses the ground plane (at or above horizon, or beyond max range)",
        )

    east, north = float(hit[0]), float(hit[1])
    lat, lon = cam.enu_to_latlon(east, north)
    g_range = cam.ground_range(hit)

    # --- real-world width, from the two bottom corners ----------------------
    width_m = 0.0
    left = cam.ray_ground_hit(x1, y2)
    right = cam.ray_ground_hit(x2, y2)
    if left is not None and right is not None:
        width_m = float(np.linalg.norm(right - left))

    # --- real-world height, from the top edge -------------------------------
    # Follow the top-centre ray until it is as far out horizontally as the
    # object base, then read how high above the ground it has got. That
    # vertical gap is the object height.
    height_m = 0.0
    top_dir = cam.pixel_to_ray(base_u, y1)
    horizontal_speed = math.hypot(top_dir[0], top_dir[1])
    if horizontal_speed > 1e-9:
        t_top = g_range / horizontal_speed
        z_top = cam.altitude_m + t_top * top_dir[2]
        height_m = max(0.0, float(z_top))

    # A near-nadir view looks down on rooftops, so the bbox says almost nothing
    # about height and the maths above turns unstable. Fall back to the ground
    # footprint, which is the honest amount of information the image carries.
    if cam.pitch_deg > 75.0 or horizontal_speed <= 1e-9:
        far = cam.ray_ground_hit(base_u, y1)
        depth_m = float(np.linalg.norm(far - hit)) if far is not None else width_m
        height_m = max(width_m, depth_m) * 0.5

    # Keep the answer inside the realm of the physically possible. See the note
    # on MAX_PLAUSIBLE_HEIGHT_M -- this catches a mismatched camera pose rather
    # than any error in the geometry above.
    height_m = min(height_m, MAX_PLAUSIBLE_HEIGHT_M)

    return GroundFix(
        valid=True,
        east_m=east,
        north_m=north,
        lat=lat,
        lon=lon,
        ground_range_m=g_range,
        slant_range_m=cam.slant_range(hit),
        bearing_deg=cam.bearing_from_camera(hit),
        width_m=width_m,
        height_m=height_m,
    )


def ground_footprint(cam: CameraModel) -> list[dict]:
    """Project the four image corners onto the ground.

    Draw these on the map and you get the camera coverage polygon -- the patch
    of world the photo actually sees. Corners whose rays escape above the
    horizon are dropped, so an oblique shot yields a partial (open) footprint
    rather than a confidently wrong closed one.
    """
    w, h = cam.image_width, cam.image_height
    # near-left, near-right, far-right, far-left
    corners = [(0, h), (w, h), (w, 0), (0, 0)]

    out: list[dict] = []
    for u, v in corners:
        hit = cam.ray_ground_hit(u, v)
        if hit is None:
            continue
        lat, lon = cam.enu_to_latlon(float(hit[0]), float(hit[1]))
        out.append({
            "lat": lat,
            "lon": lon,
            "east_m": float(hit[0]),
            "north_m": float(hit[1]),
        })
    return out
