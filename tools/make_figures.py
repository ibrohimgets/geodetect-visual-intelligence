#!/usr/bin/env python
"""
Render the figures used in the README, straight from the real KITTI data.

    python tools/make_figures.py

Nothing here is drawn by hand. Every figure is produced by running the actual
pipeline over the bundled drive, so if the geometry changes the pictures change
with it -- which is the only way a README figure stays honest.

Produces, into docs/images/:

    detections.jpg           what YOLO found, with each object's measured range
    tracking.jpg             the same objects keeping their identity across frames
    lidar-projection.png     the Velodyne cloud projected into the camera frame
    bev.png                  bird's eye view of one frame, measured positions
    measured-vs-estimated.png  LiDAR range against the flat-ground estimate
    ego-motion.png           the validation: parked cars hold still while the
                             vehicle drives past them

Photographs are written as JPEG and plots as PNG, which is simply the right
format for each: JPEG would smear the single-pixel LiDAR returns, and PNG
triples the size of a street scene for no visible gain.

Needs the KITTI drive under data/kitti (see the README for the download).
scene-3d.png is captured from the running app (Export -> Screenshot with the
3D pane in front), not produced here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Circle, Rectangle, Wedge  # noqa: E402
from PIL import Image, ImageDraw                     # noqa: E402

from app import kitti                                # noqa: E402
from app.config import KITTI_DIR, MODELS_DIR, MODEL_WEIGHTS  # noqa: E402
from app.detector import Detector                    # noqa: E402
from app.geo import CameraModel, project_detection   # noqa: E402

OUT = ROOT / "docs" / "images"

# Match the application's dark palette so the figures and the UI look related.
BG = "#0b0c0e"
PANEL = "#131519"
INK = "#d6dae1"
INK_2 = "#98a0ac"
INK_3 = "#6d7480"
GRID = "#24282f"
ACCENT = "#4c8dff"
OK = "#3fb950"
WARN = "#d29922"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_3, "ytick.color": INK_3,
    "axes.edgecolor": GRID, "grid.color": GRID,
    "font.family": "monospace", "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "figure.dpi": 130,
})


def load():
    """The sequence, the detector, and per-frame detections with LiDAR fixes."""
    sequences = kitti.discover(KITTI_DIR)
    if not sequences:
        print(f"No KITTI data under {KITTI_DIR}.", file=sys.stderr)
        print("See the README for the download command.", file=sys.stderr)
        raise SystemExit(1)

    sequence = sequences[0]
    detector = Detector(weights=MODEL_WEIGHTS, models_dir=MODELS_DIR)
    print(f"  sequence {sequence.name}: {sequence.frame_count} frames")
    return sequence, detector


# The same twelve hues the application uses, so a car is the same colour in the
# figures as it is on screen.
CLASS_COLOURS = [
    "#4cc2ff", "#ffb454", "#57d97f", "#ff7ab8", "#b48cff", "#ff8a4c",
    "#3fd6c8", "#ff6b6b", "#b5e05a", "#6aa9ff", "#ff7a7a", "#ffd24c",
]


def _hex(value: str) -> tuple[int, int, int]:
    n = int(value.lstrip("#"), 16)
    return (n >> 16) & 255, (n >> 8) & 255, n & 255


def _font(size: int = 13):
    """A real TrueType face if one is available, else PIL's bitmap default."""
    from PIL import ImageFont
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(image, detections, fixes=None, *, by_track=False, font_size=13):
    """Draw detection boxes the way the application draws them.

    Labels sit above the box, or inside it when the box is near the top edge,
    which is the same rule the canvas overlay uses -- the figures should look
    like the tool, not like a different program.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _font(font_size)

    for i, det in enumerate(detections):
        fix = fixes[i] if fixes else None
        key = det.track_id if (by_track and det.track_id is not None) else det.class_id
        colour = _hex(CLASS_COLOURS[key % len(CLASS_COLOURS)])

        draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=colour, width=2)
        draw.rectangle([det.x1, det.y1, det.x2, det.y2], fill=colour + (26,))

        name = (f"{det.class_name}_{det.track_id:02d}"
                if by_track and det.track_id is not None else det.class_name)
        parts = [name, f"{det.confidence * 100:.0f}%"]
        if fix is not None and fix.ok:
            parts.append(f"{fix.distance_m:.1f}m")
        label = "  ".join(parts)

        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        pad = 3
        ly = det.y1 - th - pad * 2
        if ly < 0:
            ly = det.y1
        draw.rectangle([det.x1, ly, det.x1 + tw + pad * 2, ly + th + pad * 2],
                       fill=colour)
        draw.text((det.x1 + pad, ly + pad), label, fill=(11, 12, 14), font=font)

    return canvas


# --------------------------------------------------------------------------
# 0. Detection results
# --------------------------------------------------------------------------


def fig_detections(sequence, detector, frame_index: int = 0) -> None:
    """What the detector found, and how far away each thing actually is.

    The plain object-detection picture, except every box also carries a range
    that came off the LiDAR rather than out of the network.
    """
    frame = sequence.frames[frame_index]
    image = Image.open(frame.image_path).convert("RGB")
    points = frame.lidar()

    detections = detector.detect(np.asarray(image), confidence=0.35).detections
    fixes = kitti.measure_detections(
        [(d.x1, d.y1, d.x2, d.y2) for d in detections],
        points, sequence.calib, frame.pose(),
    )
    canvas = annotate(image, detections, fixes)

    fig, ax = plt.subplots(figsize=(12.4, 4.2))
    ax.imshow(canvas)
    ax.set_axis_off()
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    measured = sum(1 for f in fixes if f.ok)
    ax.set_title(
        f"YOLOv8 detections  ·  {summary}  ·  {measured}/{len(detections)} ranged by LiDAR",
        loc="left", pad=8,
    )
    fig.savefig(OUT / "detections.jpg", bbox_inches="tight", pad_inches=0.12,
                pil_kwargs={"quality": 92, "optimize": True})
    plt.close(fig)
    print(f"  detections.jpg  ({len(detections)} objects)")


def fig_tracking(sequence, detector) -> None:
    """The same objects, four moments apart, keeping their identities.

    Colour is keyed to the track id rather than the class here, which is the
    whole point: car_03 stays the same colour from the first frame to the last
    even as the vehicle drives past it.
    """
    result = kitti.process_sequence(
        sequence, detector, ROOT / "data" / "uploads", confidence=0.35,
    )
    n = len(result.frames)
    picks = [0, n // 3, (2 * n) // 3, n - 1]

    # Two columns rather than a single stack: a KITTI frame is 3.3:1, so four
    # of them in one column leaves the labels too small to read and the figure
    # too tall for a README.
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 4.7), constrained_layout=True)
    for ax, index in zip(axes.ravel(), picks):
        frame = result.frames[index]
        source = sequence.frames[index]
        image = Image.open(source.image_path)
        ax.imshow(annotate(image, frame.detections, frame.fixes,
                           by_track=True, font_size=17))
        ax.set_axis_off()
        ids = [d.track_id for d in frame.detections if d.track_id is not None]
        ax.set_title(
            f"t = {frame.time_s:4.1f} s   {frame.speed_kmh:.0f} km/h   "
            f"ids {sorted(ids)}",
            loc="left", fontsize=8.5, pad=3, color=INK_2,
        )

    fig.suptitle(
        f"ByteTrack identity across the drive  ·  "
        f"{len(result.track_ids())} tracks over {n} frames",
        x=0.004, ha="left", fontsize=11.5, fontweight="bold",
    )
    fig.savefig(OUT / "tracking.jpg", pad_inches=0.14,
                pil_kwargs={"quality": 90, "optimize": True})
    plt.close(fig)
    print(f"  tracking.jpg  ({len(result.track_ids())} tracks)")


# --------------------------------------------------------------------------
# 1. LiDAR projected into the camera frame
# --------------------------------------------------------------------------


def fig_lidar_projection(sequence, detector, frame_index: int = 0) -> None:
    """The point cloud drawn over the photograph it was captured with.

    This is the figure that shows the calibration is right: the returns should
    hug the cars and the kerb, stop at the sky, and thin out with distance.
    """
    frame = sequence.frames[frame_index]
    calib = sequence.calib
    pose = frame.pose()

    image = Image.open(frame.image_path).convert("RGB")
    points = frame.lidar()
    uv, depth, inside = calib.project_lidar(points)

    detections = detector.detect(np.asarray(image), confidence=0.35).detections
    boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
    fixes = kitti.measure_detections(boxes, points, calib, pose)

    # Draw the cloud with PIL: 17k individual points is far faster this way
    # than as a matplotlib scatter, and stays pixel-exact.
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    cmap = plt.get_cmap("turbo")

    vu, vd = uv[inside], depth[inside]
    order = np.argsort(-vd)          # far points first, so near ones draw on top
    for (u, v), z in zip(vu[order], vd[order]):
        t = min(1.0, max(0.0, (z - 3.0) / 45.0))
        r, g, b, _ = cmap(1.0 - t)
        draw.point((u, v), fill=(int(r * 255), int(g * 255), int(b * 255)))

    for det, fix in zip(detections, fixes):
        colour = (63, 233, 120) if fix.ok else (248, 81, 73)
        draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=colour, width=2)
        label = (f"{det.class_name} {fix.distance_m:.1f}m ({fix.point_count}pts)"
                 if fix.ok else f"{det.class_name} no return")
        ty = max(0, det.y1 - 12)
        draw.rectangle([det.x1, ty, det.x1 + 7 * len(label), ty + 12], fill=(11, 12, 14))
        draw.text((det.x1 + 2, ty + 1), label, fill=colour)

    fig, (ax, cax) = plt.subplots(
        2, 1, figsize=(12.4, 4.6), height_ratios=[16, 1],
        gridspec_kw={"hspace": 0.28},
    )
    ax.imshow(canvas)
    ax.set_axis_off()
    ax.set_title(
        f"Velodyne HDL-64E projected into camera 2  ·  {len(points):,} returns, "
        f"{int(inside.sum()):,} land in frame",
        loc="left", pad=8,
    )

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    cax.imshow(gradient, aspect="auto", cmap=plt.get_cmap("turbo_r"),
               extent=[3, 48, 0, 1])
    cax.set_yticks([])
    cax.set_xlabel("measured range (m)", labelpad=2)
    cax.tick_params(length=2)

    fig.savefig(OUT / "lidar-projection.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("  lidar-projection.png")


# --------------------------------------------------------------------------
# 2. Bird's eye view
# --------------------------------------------------------------------------


def fig_bev(sequence, detector, frame_index: int = 0) -> None:
    """Looking straight down on one frame: where the pipeline thinks things are.

    The camera only sees an 81 degree wedge, so objects appear inside it while
    the LiDAR returns wrap all the way around the vehicle. That contrast is the
    point of the figure.
    """
    frame = sequence.frames[frame_index]
    calib = sequence.calib
    pose = frame.pose()

    points = frame.lidar()
    image = np.asarray(Image.open(frame.image_path).convert("RGB"))
    detections = detector.detect(image, confidence=0.35).detections
    fixes = kitti.measure_detections(
        [(d.x1, d.y1, d.x2, d.y2) for d in detections], points, calib, pose,
    )

    # LiDAR in the vehicle's own frame: x forward, y left.
    ground_z = kitti.ground_level_velo(points)
    above = points[(points[:, 2] > ground_z + 0.2) & (points[:, 2] < ground_z + 3.0)]
    road = points[np.abs(points[:, 2] - ground_z) <= 0.2]

    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    # Road surface first, as a faint carpet; then everything standing on it,
    # coloured by height so kerbs, cars and walls separate.
    ax.scatter(-road[::4, 1], road[::4, 0], s=0.22, c="#39414d", linewidths=0)
    ax.scatter(-above[::2, 1], above[::2, 0], s=0.55,
               c=above[::2, 2] - ground_z, cmap="viridis",
               vmin=0, vmax=2.6, linewidths=0, alpha=0.95)

    # Camera field of view, swept around the optical axis.
    axis = calib.camera_axis_in_imu()
    axis_deg = math.degrees(math.atan2(axis[1], axis[0]))
    half = calib.hfov_deg / 2
    ax.add_patch(Wedge((0, 0), 44, 90 - axis_deg - half, 90 - axis_deg + half,
                       facecolor=ACCENT, alpha=0.07, edgecolor=ACCENT,
                       linewidth=0.9, linestyle="--", zorder=1))

    for r in (10, 20, 30, 40):
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor=GRID,
                            linewidth=0.7, zorder=2))
        # clip_on keeps a ring label from escaping into the figure margin when
        # its radius falls outside the framed area.
        ax.text(0.6, r - 1.5, f"{r} m", color=INK_3, fontsize=7, zorder=2,
                clip_on=True)

    ax.add_patch(Rectangle((-0.9, -1.4), 1.8, 4.2, facecolor=ACCENT,
                           edgecolor="#ffffff", linewidth=0.8, alpha=0.95, zorder=6))
    ax.text(0, -2.9, "ego", color=ACCENT, ha="center", fontsize=8, zorder=6)

    # Place the objects, then lay out their labels so they do not sit on top of
    # one another -- parked cars cluster, and unreadable labels would waste the
    # only figure that shows measured position directly.
    placed = []
    for det, fix in zip(detections, fixes):
        if not fix.ok:
            continue
        # measure_detections returns world ENU; rotate back into the body frame
        # so the figure is vehicle-centred rather than north-up.
        body = pose.rotation_to_enu().T @ np.array([fix.east_m, fix.north_m, 0.0])
        placed.append((float(-body[1]), float(body[0]), det.class_name, fix))

    for x, y, name, fix in placed:
        w = max(1.2, fix.width_m)
        ax.add_patch(Rectangle((x - w / 2, y - w / 2), w, w,
                               facecolor=OK, alpha=0.28, edgecolor=OK,
                               linewidth=1.3, zorder=7))

    # Greedy vertical de-overlap, nearest object first.
    taken: list[tuple[float, float]] = []
    for x, y, name, fix in sorted(placed, key=lambda p: p[1]):
        w = max(1.2, fix.width_m)
        ly = y + w / 2 + 1.4
        while any(abs(ly - py) < 3.4 and abs(x - px) < 9.0 for px, py in taken):
            ly += 3.4
        taken.append((x, ly))
        ax.annotate(
            f"{name}  {fix.distance_m:.1f} m",
            xy=(x, y + w / 2), xytext=(x, ly),
            color=OK, ha="center", fontsize=7.5, zorder=8,
            bbox=dict(boxstyle="square,pad=0.22", facecolor=BG,
                      edgecolor=OK, linewidth=0.6),
            arrowprops=dict(arrowstyle="-", color=OK, linewidth=0.6,
                            shrinkA=0, shrinkB=2),
        )

    # Frame the content rather than the whole cloud.
    far = max((y for _, y, _, _ in placed), default=25) + 12
    ax.set_xlim(-24, 24)
    ax.set_ylim(-11, max(30, far))
    ax.set_aspect("equal")
    ax.set_xlabel("left  <-   metres   ->  right")
    ax.set_ylabel("metres ahead")
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.set_title(
        f"Bird's eye view  ·  {len(placed)} objects placed from LiDAR\n"
        f"the cloud wraps 360°, the camera sees {calib.hfov_deg:.0f}°",
        loc="left", pad=10,
    )

    fig.savefig(OUT / "bev.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  bev.png")


# --------------------------------------------------------------------------
# 3. Measured against estimated
# --------------------------------------------------------------------------


def fig_measured_vs_estimated(sequence, detector) -> None:
    """What the flat-ground assumption costs you when depth is available.

    Both numbers come from the same detection box. One reads the range off the
    LiDAR; the other assumes the object stands on a flat plane at a known camera
    height. The gap between them is the error the monocular path carries
    silently, and the reason the UI labels the two differently.
    """
    measured: list[float] = []
    estimated: list[float] = []
    classes: list[str] = []

    for frame in sequence.frames:
        pose = frame.pose()
        points = frame.lidar()
        ground_z = kitti.ground_level_velo(points)
        image = np.asarray(Image.open(frame.image_path).convert("RGB"))
        detections = detector.detect(image, confidence=0.4).detections
        if not detections:
            continue

        fixes = kitti.measure_detections(
            [(d.x1, d.y1, d.x2, d.y2) for d in detections],
            points, sequence.calib, pose,
        )
        cp = kitti.camera_pose_for_fallback(sequence.calib, pose, ground_z)
        cam = CameraModel(
            image_width=sequence.calib.image_size[0],
            image_height=sequence.calib.image_size[1],
            hfov_deg=cp["hfov_deg"], altitude_m=cp["altitude_m"],
            pitch_deg=cp["pitch_deg"], heading_deg=cp["heading_deg"],
            lat=pose.lat, lon=pose.lon, max_range_m=200,
            fx_px=cp["fx_px"], fy_px=cp["fy_px"],
            cx_px=cp["cx_px"], cy_px=cp["cy_px"],
        )

        for det, fix in zip(detections, fixes):
            if not fix.ok:
                continue
            gp = project_detection(cam, det.x1, det.y1, det.x2, det.y2)
            if not gp.valid:
                continue
            measured.append(fix.distance_m)
            estimated.append(gp.ground_range_m)
            classes.append(det.class_name)

    measured_a = np.array(measured)
    estimated_a = np.array(estimated)
    error = estimated_a - measured_a

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    lim = max(measured_a.max(), estimated_a.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=INK_3, linewidth=0.9, linestyle="--",
            label="perfect agreement")
    ax.scatter(measured_a, estimated_a, s=18, c=ACCENT, alpha=0.7,
               edgecolors="none", label=f"{len(measured_a)} detections")
    ax.set_xlabel("LiDAR measured range (m)")
    ax.set_ylabel("flat-ground estimate (m)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK_2, fontsize=8)
    ax.set_title("Estimate vs measurement", loc="left", pad=8)

    ax2.axhline(0, color=INK_3, linewidth=0.9, linestyle="--")
    ax2.scatter(measured_a, error, s=18, c=WARN, alpha=0.75, edgecolors="none")
    ax2.set_xlabel("LiDAR measured range (m)")
    ax2.set_ylabel("estimate − measured (m)")
    ax2.grid(alpha=0.25, linewidth=0.5)
    ax2.set_title(
        f"Error  ·  median {np.median(error):+.1f} m  ·  "
        f"90th pct {np.percentile(np.abs(error), 90):.1f} m",
        loc="left", pad=8,
    )

    fig.suptitle(
        "The flat-ground assumption, measured against the LiDAR it replaces",
        x=0.012, ha="left", fontsize=11.5, fontweight="bold", y=1.02,
    )
    fig.savefig(OUT / "measured-vs-estimated.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"  measured-vs-estimated.png  ({len(measured_a)} pairs, "
          f"median error {np.median(error):+.2f} m)")


# --------------------------------------------------------------------------
# 4. Ego-motion validation
# --------------------------------------------------------------------------


def fig_ego_motion(sequence, detector) -> None:
    """The end-to-end check, as a picture.

    Track objects across the drive and plot each one twice: where it sits
    relative to the vehicle, and where it sits on the Earth. The vehicle covers
    17 metres, so the relative positions sweep. The absolute ones must not --
    and that they do not is what says the calibration, the LiDAR projection, the
    IMU rotation and the yaw convention are all right together.
    """
    result = kitti.process_sequence(
        sequence, detector, ROOT / "data" / "uploads", confidence=0.35,
    )

    tracks: dict[int, dict] = {}
    for frame in result.frames:
        cam = CameraModel(image_width=result.width, image_height=result.height,
                          lat=frame.camera["lat"], lon=frame.camera["lon"])
        for det, fix in zip(frame.detections, frame.fixes):
            if det.track_id is None or not fix.ok:
                continue
            entry = tracks.setdefault(det.track_id, {"rel": [], "abs": [], "cls": det.class_name})
            entry["rel"].append((fix.east_m, fix.north_m))
            entry["abs"].append(cam.enu_to_latlon(fix.east_m, fix.north_m))

    persistent = {k: v for k, v in tracks.items() if len(v["rel"]) >= 6}

    first, last = sequence.frames[0].pose(), sequence.frames[-1].pose()
    lat0, lon0 = first.lat, first.lon
    to_m = lambda lat, lon: (                                    # noqa: E731
        (lon - lon0) * 111320.0 * math.cos(math.radians(lat0)),
        (lat - lat0) * 111132.0,
    )
    ego = [to_m(f.pose().lat, f.pose().lon) for f in sequence.frames]
    ego_dist = math.hypot(ego[-1][0] - ego[0][0], ego[-1][1] - ego[0][1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0))
    colours = plt.get_cmap("tab10")

    for i, (tid, entry) in enumerate(sorted(persistent.items())):
        c = colours(i % 10)
        rel = np.array(entry["rel"])
        ax1.plot(rel[:, 0], rel[:, 1], "-o", color=c, markersize=2.4,
                 linewidth=1.2, label=f"{entry['cls']}_{tid:02d}")

        ab = np.array([to_m(la, lo) for la, lo in entry["abs"]])
        ax2.plot(ab[:, 0], ab[:, 1], "-o", color=c, markersize=2.4, linewidth=1.2)

    ax1.plot(0, 0, marker="s", color=ACCENT, markersize=9, zorder=5)
    ax1.annotate("camera", xy=(0, 0), xytext=(10, -10),
                 textcoords="offset points", color=ACCENT, fontsize=8, zorder=5)
    ax1.set_title("Relative to the vehicle\nobjects sweep past", loc="left", pad=8)
    ax1.set_xlabel("east of camera (m)"); ax1.set_ylabel("north of camera (m)")
    ax1.grid(alpha=0.25, linewidth=0.5); ax1.set_aspect("equal")
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK_2,
               fontsize=7, loc="lower right", framealpha=0.95)

    ego_a = np.array(ego)
    ax2.plot(ego_a[:, 0], ego_a[:, 1], "-", color=ACCENT, linewidth=2.0,
             label=f"vehicle path, {ego_dist:.1f} m")
    ax2.plot(ego_a[0, 0], ego_a[0, 1], marker="s", color=ACCENT, markersize=8)
    ax2.set_title("On the Earth (WGS84)\nthe same objects hold still", loc="left", pad=8)
    ax2.set_xlabel("east of start (m)"); ax2.set_ylabel("north of start (m)")
    ax2.grid(alpha=0.25, linewidth=0.5); ax2.set_aspect("equal")
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK_2,
               fontsize=7, loc="lower right", framealpha=0.95)

    spreads = []
    for entry in persistent.values():
        ab = np.array([to_m(la, lo) for la, lo in entry["abs"]])
        spreads.append(math.hypot(ab[:, 0].ptp(), ab[:, 1].ptp()))

    fig.suptitle(
        f"Ego-motion cancellation  ·  vehicle travels {ego_dist:.1f} m  ·  "
        f"tracked objects hold to a median {np.median(spreads):.2f} m",
        x=0.012, ha="left", fontsize=11.5, fontweight="bold", y=1.0,
    )
    fig.savefig(OUT / "ego-motion.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"  ego-motion.png  (ego {ego_dist:.1f} m, "
          f"median object spread {np.median(spreads):.2f} m)")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sequence, detector = load()
    fig_detections(sequence, detector)
    fig_tracking(sequence, detector)
    fig_lidar_projection(sequence, detector)
    fig_bev(sequence, detector)
    fig_measured_vs_estimated(sequence, detector)
    fig_ego_motion(sequence, detector)
    print(f"\nWrote figures to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
