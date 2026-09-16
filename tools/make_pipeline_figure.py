#!/usr/bin/env python
"""
Render the architecture figure for the README.

    python tools/make_pipeline_figure.py     ->  docs/images/pipeline.png

The layout is the one multi-sensor fusion papers use: one lane per modality
running left to right, the lanes converging into a single shared
representation, and that representation fanning out to the things built on top
of it. BEVFusion (Liu et al., ICRA 2023, Fig. 2) is the specific reference for
the visual language -- white ground, one hue per modality, geometric glyphs
instead of labelled rectangles, heavy black arrows, and real data at the inputs
with real results at the outputs.

Two things were added to that vocabulary, because this pipeline has a claim to
make that BEVFusion's does not. A **dashed slate lane** carries the monocular
fallback: same detections, no depth sensor, a flat-ground assumption instead.
And **violet chips** hang the fixed sensor constants -- calibration, OXTS --
off the stages that consume them, so the figure says where each number enters.

Everything with data in it is real data. The camera thumbnail is a KITTI frame,
the point cloud is that frame's Velodyne sweep, the map is the street the drive
was recorded on with every measured object at its computed coordinates, and the
assistant card runs a real spatial query against the real scene graph.

Needs the KITTI drive under data/kitti (see the README) and, for the map card,
a network connection. Without the network it falls back to plotting the same
coordinates on a plain ground, and says so.
"""

from __future__ import annotations

import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont                        # noqa: E402

from app import kitti, scene_graph                                 # noqa: E402
from app.config import KITTI_DIR, MODELS_DIR, MODEL_WEIGHTS        # noqa: E402
from app.detector import Detector                                  # noqa: E402
from app.geo import CameraModel                                    # noqa: E402
from app.main import _detection_out, _world_from_ground_plane, _world_from_lidar  # noqa: E402
from make_figures import CLASS_COLOURS                             # noqa: E402

OUT = ROOT / "docs" / "images"

# --------------------------------------------------------------------------
# Palette, sampled from the reference figure so the register matches exactly
# --------------------------------------------------------------------------

PAPER = "#feffff"
INK = "#000000"

AMBER = ("#f8b62d", "#fef0d8")    # the camera lane: what the thing is
BLUE = ("#0081cc", "#d3e5f5")     # the LiDAR lane: where it actually is
TEAL = ("#00adba", "#d6eef1")     # the shared world representation
SLATE = ("#8c96a3", "#eceff2")    # the fallback lane: assumed, not measured
VIOLET = ("#7b61c9", "#eae4f8")   # fixed sensor constants feeding a stage

# One Arial-ish grotesque throughout, like the reference. DejaVu is the
# fallback on machines without Arial; it is wider, so the figure breathes less.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "text.color": INK,
})

# The canvas. 8 units to the inch, so a point is 1/9 of a unit and the type
# sizes below can be reasoned about against the geometry.
W, H = 109.0, 31.0
UNITS_PER_INCH = 8.0

F_TITLE = 12.0     # a module's name
F_SUB = 9.6        # the qualifier under it
F_HEAD = 13.5      # the group header over the output cards
F_CHIP = 9.3       # sensor-constant chips and provenance badges

LW = 2.2           # glyph outline
LW_THIN = 1.3


# --------------------------------------------------------------------------
# Glyph vocabulary
# --------------------------------------------------------------------------


def funnel(ax, cx, cy, w, h, colour, dashed=False):
    """Two mirrored trapezoids: the reference's 'encoder' glyph.

    Used wherever a stage consumes something rich and emits something
    smaller -- the detector, and the geo-referencing step.
    """
    stroke, fill = colour
    half, waist, gap = w / 2, h * 0.40, w * 0.07
    style = (0, (4, 2)) if dashed else "solid"
    for sign in (-1, 1):
        ax.add_patch(Polygon(
            [(cx + sign * half, cy + h / 2), (cx + sign * gap, cy + waist / 2),
             (cx + sign * gap, cy - waist / 2), (cx + sign * half, cy - h / 2)],
            closed=True, fc=fill, ec=stroke, lw=LW, ls=style,
            joinstyle="miter", zorder=3))


def iso_box(ax, cx, cy, w, h, depth, layers, nx=3, ny=1):
    """An axonometric box, optionally stacked out of coloured layers.

    A single layer is a slab -- one value. Several layers stacked is the
    reference's way of showing a record made of parts that stay separable,
    which is exactly what an object record is here.
    """
    dx, dy = depth * 0.62, depth * 0.52
    x0, y0 = cx - w / 2 - dx / 2, cy - h / 2 - dy / 2
    lh = h / len(layers)

    for i, (colour, dashed) in enumerate(layers):
        stroke, fill = colour
        ls = (0, (3.5, 2)) if dashed else "solid"
        y = y0 + i * lh
        kw = dict(fc=fill, ec=stroke, lw=LW, ls=ls, joinstyle="miter", zorder=3)
        ax.add_patch(Polygon(                                    # front face
            [(x0, y), (x0 + w, y), (x0 + w, y + lh), (x0, y + lh)], **kw))
        ax.add_patch(Polygon(                                    # right face
            [(x0 + w, y), (x0 + w + dx, y + dy),
             (x0 + w + dx, y + lh + dy), (x0 + w, y + lh)], **kw))
        if i == len(layers) - 1:                                 # top face
            ax.add_patch(Polygon(
                [(x0, y + lh), (x0 + w, y + lh),
                 (x0 + w + dx, y + lh + dy), (x0 + dx, y + lh + dy)], **kw))

        for k in range(1, nx):                                   # cell divisions
            x = x0 + w * k / nx
            ax.plot([x, x], [y, y + lh], color=stroke, lw=LW_THIN, zorder=4)
        if i == len(layers) - 1:
            for k in range(1, nx):
                x = x0 + w * k / nx
                ax.plot([x, x + dx], [y + lh, y + lh + dy],
                        color=stroke, lw=LW_THIN, zorder=4)


def card_stack(ax, cx, cy, w, h, colour, n=3, marks=()):
    """Offset cards, back to front: a set of things of the same kind.

    `marks` draws little boxes on the front card, so 'Detections' looks like
    detections rather than like any other stack of feature maps.
    """
    stroke, fill = colour
    step = min(w, h) * 0.13
    x0, y0 = cx - w / 2 - step * (n - 1) / 2, cy - h / 2 - step * (n - 1) / 2
    for i in reversed(range(n)):
        x, y = x0 + i * step, y0 + i * step
        ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=stroke, lw=LW,
                               joinstyle="miter", zorder=3 + (n - i)))
        if i == 0:
            for mx, my, mw, mh in marks:
                ax.add_patch(Rectangle(
                    (x + mx * w, y + my * h), mw * w, mh * h,
                    fc="none", ec=INK, lw=LW_THIN, zorder=9))


def tile(ax, cx, cy, w, h, colour, dashed=False):
    """A rounded square holding a pictogram: an operation, not a data object."""
    stroke, fill = colour
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.5",
        fc=fill, ec=stroke, lw=LW,
        ls=(0, (4, 2)) if dashed else "solid", zorder=3))


def pictogram_projection(ax, cx, cy, s):
    """A camera at a point, its rays, and returns landing on the image plane."""
    apex = (cx - s * 0.95, cy)
    for dy in (-s * 0.62, s * 0.62):
        ax.plot([apex[0], cx + s * 0.15], [apex[1], cy + dy],
                color=INK, lw=LW_THIN, zorder=6)
    ax.plot([cx + s * 0.15, cx + s * 0.15], [cy - s * 0.62, cy + s * 0.62],
            color=INK, lw=LW * 0.9, zorder=6)
    ax.plot(*apex, marker="o", ms=3.2, color=INK, zorder=6)
    rng = np.random.default_rng(7)
    xs = cx + s * 0.45 + rng.random(11) * s * 0.55
    ys = cy + (rng.random(11) - 0.5) * s * 1.25
    ax.scatter(xs, ys, s=3.0, color=INK, zorder=6)


def pictogram_track(ax, cx, cy, s):
    """Three positions of one object, linked: identity carried across frames."""
    xs = [cx - s * 0.8, cx, cx + s * 0.8]
    ys = [cy - s * 0.42, cy + s * 0.1, cy + s * 0.44]
    ax.plot(xs, ys, color=INK, lw=LW_THIN, ls=(0, (2.5, 2)), zorder=6)
    for i, (x, y) in enumerate(zip(xs, ys)):
        side = s * (0.34 + i * 0.08)
        ax.add_patch(Rectangle((x - side / 2, y - side / 2), side, side,
                               fc="none", ec=INK, lw=LW_THIN, zorder=6))


def pictogram_ground_ray(ax, cx, cy, s):
    """A ray from the camera meeting the assumed ground plane."""
    ax.plot([cx - s, cx + s], [cy - s * 0.55, cy - s * 0.55],
            color=INK, lw=LW * 0.9, zorder=6)
    ax.plot([cx - s * 0.85, cx + s * 0.55], [cy + s * 0.75, cy - s * 0.55],
            color=INK, lw=LW_THIN, zorder=6)
    ax.plot(cx + s * 0.55, cy - s * 0.55, marker="o", ms=4.0,
            mfc=PAPER, mec=INK, mew=1.3, zorder=7)
    for k in (-0.55, -0.15, 0.25):                       # hatching under the plane
        ax.plot([cx + k * s, cx + (k - 0.2) * s],
                [cy - s * 0.55, cy - s * 0.85], color=INK, lw=0.9, zorder=6)


def block_arrow(ax, x0, y0, x1, y1, shaft=0.62, head_w=1.9, head_l=1.5,
                colour=INK, dashed=False, z=8):
    """The reference's chunky filled arrow: a rectangle plus a triangle."""
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    bx, by = x1 - ux * head_l, y1 - uy * head_l

    if dashed:
        ax.plot([x0, bx], [y0, by], color=colour, lw=shaft * 6.0,
                ls=(0, (1.5, 1.1)), solid_capstyle="butt", zorder=z)
    else:
        ax.add_patch(Polygon(
            [(x0 + px * shaft / 2, y0 + py * shaft / 2),
             (bx + px * shaft / 2, by + py * shaft / 2),
             (bx - px * shaft / 2, by - py * shaft / 2),
             (x0 - px * shaft / 2, y0 - py * shaft / 2)],
            closed=True, fc=colour, ec="none", zorder=z))
    ax.add_patch(Polygon(
        [(x1, y1), (bx + px * head_w / 2, by + py * head_w / 2),
         (bx - px * head_w / 2, by - py * head_w / 2)],
        closed=True, fc=colour, ec="none", zorder=z))


def label(ax, cx, y, title, sub, above, colour=INK):
    """A module's two-line caption, set away from the middle of the figure."""
    gap = 1.28
    if above:
        ax.text(cx, y + gap, title, ha="center", va="bottom",
                fontsize=F_TITLE, color=colour)
        ax.text(cx, y, sub, ha="center", va="bottom",
                fontsize=F_SUB, color=colour)
    else:
        ax.text(cx, y, title, ha="center", va="top",
                fontsize=F_TITLE, color=colour)
        ax.text(cx, y - gap, sub, ha="center", va="top",
                fontsize=F_SUB, color=colour)


def chip(ax, cx, cy, w, h, lines, colour=VIOLET):
    """A fixed sensor constant, hung off the stage that consumes it."""
    stroke, fill = colour
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.42",
        fc=fill, ec=stroke, lw=LW_THIN, zorder=3))
    step = h / (len(lines) + 1)
    for i, text in enumerate(lines):
        ax.text(cx, cy + h / 2 - step * (i + 1), text, ha="center",
                va="center", fontsize=F_CHIP, color="#3a2f66", zorder=6)


def badge(ax, cx, cy, text, colour, italic=False):
    """MEASURED / estimated: which of the two paths produced this number."""
    stroke, fill = colour
    ax.text(cx, cy, text, ha="center", va="center", fontsize=F_CHIP,
            color=stroke, style="italic" if italic else "normal",
            fontweight="normal" if italic else "bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.34", fc=fill, ec=stroke, lw=1.1))


def thumb(ax, cx, cy, w, image, stack=1):
    """A real photograph, framed the way the reference frames its inputs."""
    arr = np.asarray(image)
    h = w * arr.shape[0] / arr.shape[1]
    step = w * 0.022
    x0, y0 = cx - w / 2 - step * (stack - 1) / 2, cy - h / 2 - step * (stack - 1) / 2
    for i in reversed(range(stack)):
        x, y = x0 + i * step, y0 + i * step
        if i:
            ax.add_patch(Rectangle((x, y), w, h, fc=PAPER, ec=INK,
                                   lw=LW_THIN, zorder=3 + (stack - i)))
        else:
            ax.imshow(arr, extent=(x, x + w, y, y + h), zorder=3 + stack,
                      interpolation="antialiased")
            ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=INK,
                                   lw=LW_THIN, zorder=20))
    return h


def result_card(ax, cx, cy, w, image, caption, img_h, cap_h):
    """An output card: the result, then its name on a tinted bar beneath it."""
    x0 = cx - w / 2
    y_cap = cy - (img_h + cap_h) / 2
    ax.imshow(np.asarray(image), extent=(x0, x0 + w, y_cap + cap_h,
                                         y_cap + cap_h + img_h),
              zorder=3, interpolation="antialiased")
    ax.add_patch(Rectangle((x0, y_cap), w, cap_h, fc=TEAL[1], ec="none",
                           zorder=3))
    ax.text(cx, y_cap + cap_h / 2, caption, ha="center", va="center",
            fontsize=F_TITLE, color=INK, zorder=6)
    ax.add_patch(Rectangle((x0, y_cap), w, cap_h + img_h, fc="none",
                           ec="#2b2b2b", lw=1.4, zorder=21))


# --------------------------------------------------------------------------
# The data behind the figure
# --------------------------------------------------------------------------


def _camera_of(frame, width, height) -> CameraModel:
    """The same CameraModel the /api/kitti/timeline route builds, per frame."""
    p = frame.camera
    return CameraModel(
        image_width=width, image_height=height,
        hfov_deg=p["hfov_deg"], altitude_m=p["altitude_m"],
        pitch_deg=p["pitch_deg"], heading_deg=p["heading_deg"],
        lat=p["lat"], lon=p["lon"],
        fx_px=p["fx_px"], fy_px=p["fy_px"], cx_px=p["cx_px"], cy_px=p["cy_px"],
    )


def collect():
    """Run the real pipeline over the bundled drive and keep what we draw."""
    sequences = kitti.discover(KITTI_DIR)
    if not sequences:
        print(f"No KITTI data under {KITTI_DIR}.", file=sys.stderr)
        print("See the README for the download command.", file=sys.stderr)
        raise SystemExit(1)

    sequence = sequences[0]
    detector = Detector(weights=MODEL_WEIGHTS, models_dir=MODELS_DIR)
    print(f"  sequence {sequence.name}: {sequence.frame_count} frames")

    result = kitti.process_sequence(
        sequence, detector, ROOT / "data" / "uploads", confidence=0.35)

    placed = []       # (lat, lon, class_id, measured) for every detection
    for frame in result.frames:
        cam = _camera_of(frame, result.width, result.height)
        for det, fix in zip(frame.detections, frame.fixes):
            world = (_world_from_lidar(cam, fix) if fix.ok
                     else _world_from_ground_plane(cam, det, fix.reason))
            if world.valid:
                placed.append((world.lat, world.lon, det.class_id,
                               world.source == "lidar"))

    first = result.frames[0]
    cam0 = _camera_of(first, result.width, result.height)
    detections = []
    for det, fix in zip(first.detections, first.fixes):
        world = (_world_from_lidar(cam0, fix) if fix.ok
                 else _world_from_ground_plane(cam0, det, fix.reason))
        detections.append(json.loads(_detection_out(det, world).model_dump_json()))

    graph = scene_graph.build(detections, {
        "heading_deg": cam0.heading_deg, "lat": cam0.lat, "lon": cam0.lon,
        "altitude_m": cam0.altitude_m,
    })

    track = [(f.camera["lat"], f.camera["lon"]) for f in result.frames]
    print(f"  {len(placed)} placed detections, {len(track)} camera poses, "
          f"{len(graph.nodes)} objects in frame 0")

    return {
        "sequence": sequence,
        "image": Image.open(sequence.frames[0].image_path).convert("RGB"),
        "points": sequence.frames[0].lidar(),
        "placed": placed,
        "track": track,
        "graph": graph,
        "frames": result.frame_count,
        "tracks": len(result.track_ids()),
    }


# --------------------------------------------------------------------------
# Thumbnails, drawn from that data
# --------------------------------------------------------------------------


def lidar_bev_thumb(points, width=820, height=660,
                    lateral=(-23.0, 23.0), forward=(-14.0, 23.0)):
    """The Velodyne sweep from above: white returns on black, as papers show it.

    Brightness is the return's own intensity, so the road markings and the
    number plates come out bright and the tarmac stays dark -- the picture is
    the sensor's, not a colour map laid over it.
    """
    canvas = np.zeros((height, width), dtype=np.float32)
    x, y, intensity = points[:, 0], points[:, 1], points[:, 3]

    keep = ((x > forward[0]) & (x < forward[1]) &
            (y > lateral[0]) & (y < lateral[1]))
    x, y, intensity = x[keep], y[keep], intensity[keep]

    # Image columns run left to right with +y (left in LiDAR frame) on the
    # left, and rows run top to bottom with +x (forward) at the top.
    col = ((lateral[1] - y) / (lateral[1] - lateral[0]) * (width - 1)).astype(int)
    row = ((forward[1] - x) / (forward[1] - forward[0]) * (height - 1)).astype(int)
    value = 0.46 + 0.54 * np.clip(intensity, 0.0, 1.0)

    for dr in (-1, 0, 1):                   # a fat dot survives the downscale
        for dc in (-1, 0, 1):
            r, c = np.clip(row + dr, 0, height - 1), np.clip(col + dc, 0, width - 1)
            np.maximum.at(canvas, (r, c), value)

    grey = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([grey] * 3))


# --- the map card ----------------------------------------------------------

TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
TILE_PX = 256
ZOOM = 19          # the deepest level Esri has imagery for over Karlsruhe


def _to_pixels(lat, lon, zoom=ZOOM):
    """Web Mercator, in whole-world pixels at this zoom."""
    n = TILE_PX * 2 ** zoom
    rad = math.radians(lat)
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0 * n)


def _mosaic(x0, y0, x1, y1):
    """Stitch the World Imagery tiles covering a whole-world pixel window."""
    tx0, ty0 = int(x0 // TILE_PX), int(y0 // TILE_PX)
    tx1, ty1 = int(x1 // TILE_PX), int(y1 // TILE_PX)
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * TILE_PX,
                               (ty1 - ty0 + 1) * TILE_PX), "#11141a")
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            url = TILE_URL.format(z=ZOOM, x=tx, y=ty)
            with urllib.request.urlopen(url, timeout=25) as response:
                data = response.read()
            canvas.paste(Image.open(io.BytesIO(data)).convert("RGB"),
                         ((tx - tx0) * TILE_PX, (ty - ty0) * TILE_PX))
    return canvas, tx0 * TILE_PX, ty0 * TILE_PX


def map_thumb(placed, track, aspect=2.30, height=250):
    """Every measured object on the street the drive was recorded on.

    Real imagery, real coordinates: the markers are placed by converting each
    object's computed latitude and longitude back to pixels, which means a
    mistake anywhere in the pipeline would show up here as cars in the gardens.
    """
    lats = [p[0] for p in placed] + [t[0] for t in track]
    lons = [p[1] for p in placed] + [t[1] for t in track]
    corners = [_to_pixels(lat, lon) for lat, lon in
               ((min(lats), min(lons)), (max(lats), max(lons)))]
    xs = sorted(c[0] for c in corners)
    ys = sorted(c[1] for c in corners)

    cx, cy = (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2
    half_h = max((ys[1] - ys[0]) / 2 * 1.06, 45.0)
    half_w = half_h * aspect
    x0, y0, x1, y1 = cx - half_w, cy - half_h, cx + half_w, cy + half_h

    try:
        canvas, ox, oy = _mosaic(x0, y0, x1, y1)
        grounded = True
    except Exception as exc:                          # offline, or Esri is down
        print(f"  ! map tiles unavailable ({exc.__class__.__name__}); "
              f"drawing the coordinates on a plain ground instead")
        canvas = Image.new("RGB", (int(x1 - x0), int(y1 - y0)), "#161a20")
        ox, oy = x0, y0
        grounded = False

    draw = ImageDraw.Draw(canvas, "RGBA")

    path = [(px - ox, py - oy) for px, py in
            (_to_pixels(lat, lon) for lat, lon in track)]
    if not grounded:
        for gx in range(0, canvas.size[0], 26):
            draw.line([(gx, 0), (gx, canvas.size[1])], fill=(255, 255, 255, 16))
        for gy in range(0, canvas.size[1], 26):
            draw.line([(0, gy), (canvas.size[0], gy)], fill=(255, 255, 255, 16))
    draw.line(path, fill=(76, 141, 255, 235), width=5, joint="curve")

    for lat, lon, class_id, measured in placed:
        px, py = _to_pixels(lat, lon)
        px, py = px - ox, py - oy
        colour = CLASS_COLOURS[class_id % len(CLASS_COLOURS)]
        rgb = tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
        r = 5.4
        box = [px - r, py - r, px + r, py + r]
        if measured:
            draw.ellipse(box, fill=rgb + (235,), outline=(12, 14, 18, 255))
        else:
            draw.ellipse(box, fill=None, outline=rgb + (235,), width=2)

    end = path[-1]
    draw.ellipse([end[0] - 6.5, end[1] - 6.5, end[0] + 6.5, end[1] + 6.5],
                 fill=(76, 141, 255, 255), outline=(255, 255, 255, 255), width=2)

    return canvas.resize((int(height * aspect), height), Image.LANCZOS)


# --- the assistant card ----------------------------------------------------

def _mono(size):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def assistant_thumb(graph, width=575, height=250):
    """A real question, the query it becomes, and the answer off the graph.

    The tool call below is executed, not quoted: whatever `select_objects`
    returns for this scene is what the card reports.
    """
    question = "which objects are within 30 metres?"
    hits = scene_graph.select(graph, max_distance_m=30)
    located = [n for n in graph.nodes if n.located]

    card = Image.new("RGB", (width, height), "#131519")
    draw = ImageDraw.Draw(card)
    draw.rectangle([0, 0, width - 1, 25], fill="#181b20")
    draw.text((14, 7), "ASSISTANT", font=_mono(13), fill="#98a0ac")
    draw.text((width - 96, 7), "gpt-5.4-mini", font=_mono(13), fill="#6d7480")

    y = 40
    draw.text((14, y), f'ask  "{question}"', font=_mono(15), fill="#d6dae1")

    y += 30
    draw.rectangle([14, y - 4, width - 16, y + 24], fill="#171c26",
                   outline="#24395e")
    draw.text((22, y + 3), "select_objects(max_distance_m=30)",
              font=_mono(15), fill="#4c8dff")

    y += 42
    draw.text((14, y), f"{len(hits)} of {len(located)} located objects",
              font=_mono(15), fill="#3fb950")

    y += 26
    for node in hits[:4]:
        how = "LiDAR" if node.source == "lidar" else "estimated"
        draw.text((14, y),
                  f"  {node.label:9s} {node.distance_m:5.1f} m  {node.side:<6s} {how}",
                  font=_mono(14), fill="#98a0ac")
        y += 22

    return card


def crop_to_aspect(image, aspect):
    """Trim the long side so a card's picture is the shape the card expects."""
    w, h = image.size
    if w / h > aspect:
        new_w = int(h * aspect)
        return image.crop(((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h))
    new_h = int(w / aspect)
    return image.crop((0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h))


# --------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------

# Lane centres. The camera lane on top, the LiDAR lane at the bottom, and the
# fallback threaded between them because that is where it branches from.
Y_CAM, Y_FALL, Y_LID = 24.0, 15.3, 6.8
Y_MID = 15.3

C1, C2, C3 = 24.8, 36.7, 48.6      # the three stage columns
X_BRACKET = 54.9                   # where the lanes merge
X_REC, X_GEO, X_WORLD = 61.5, 72.5, 83.0
X_FAN = 88.6
X_CARD = 98.9
CARD_W, CARD_IMG_H, CARD_CAP_H = 17.6, 7.0, 1.85

GW, GH = 8.2, 4.8                  # the nominal glyph box


def draw(data) -> None:
    fig = plt.figure(figsize=(W / UNITS_PER_INCH, H / UNITS_PER_INCH), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- inputs ----------------------------------------------------------
    img_h = thumb(ax, 8.9, Y_CAM, 15.2, data["image"], stack=3)
    label(ax, 8.9, Y_CAM - img_h / 2 - 0.95, "Camera", "1242×375, rectified",
          above=False)

    bev = lidar_bev_thumb(data["points"])
    bev_y = Y_LID + 1.5
    bev_h = thumb(ax, 8.9, bev_y, 11.2, bev)
    label(ax, 8.9, bev_y - bev_h / 2 - 0.95, "Velodyne HDL-64E",
          "≈120k returns / frame", above=False)

    # ---- camera lane -----------------------------------------------------
    funnel(ax, C1, Y_CAM, GW * 0.78, GH, AMBER)
    label(ax, C1, Y_CAM + GH / 2, "YOLOv8-n", "COCO-80 classes", above=True)

    card_stack(ax, C2, Y_CAM, GW * 0.80, GH * 0.86, AMBER, n=3, marks=(
        (0.10, 0.20, 0.30, 0.34), (0.48, 0.30, 0.22, 0.26),
        (0.72, 0.18, 0.20, 0.22)))
    label(ax, C2, Y_CAM + GH / 2, "Detections", "box · class · conf", above=True)

    tile(ax, C3, Y_CAM, GW * 0.74, GH * 0.86, AMBER)
    pictogram_track(ax, C3, Y_CAM, 1.55)
    label(ax, C3, Y_CAM + GH / 2, "ByteTrack", "persistent ids", above=True)

    # ---- LiDAR lane ------------------------------------------------------
    tile(ax, C1, Y_LID, GW * 0.74, GH * 0.86, BLUE)
    pictogram_projection(ax, C1, Y_LID, 1.65)
    label(ax, C1, Y_LID - GH / 2, "Project to Image", "velo → pixels", above=False)

    iso_box(ax, C2, Y_LID, GW * 0.72, GH * 0.62, 1.5, [(BLUE, False)], nx=4)
    label(ax, C2, Y_LID - GH / 2, "Returns in Box", "object vs behind",
          above=False)

    iso_box(ax, C3, Y_LID, GW * 0.68, GH * 0.34, 1.4, [(BLUE, False)], nx=3)
    label(ax, C3, Y_LID - GH / 2, "Measured Range", "median of returns",
          above=False)
    badge(ax, C3, Y_LID + 2.5, "MEASURED", BLUE)

    # ---- fallback lane ---------------------------------------------------
    tile(ax, C2, Y_FALL, GW * 0.74, GH * 0.80, SLATE, dashed=True)
    pictogram_ground_ray(ax, C2, Y_FALL, 1.55)
    label(ax, C2, Y_FALL - GH / 2 + 0.3, "Ground Plane",
          "ray ∩ z = 0, flat", above=False, colour="#4c545f")

    iso_box(ax, C3, Y_FALL, GW * 0.68, GH * 0.34, 1.4, [(SLATE, True)], nx=3)
    label(ax, C3, Y_FALL - GH / 2 + 0.3, "Estimated Range", "no depth sensor",
          above=False, colour="#4c545f")
    badge(ax, C3, Y_FALL + 2.2, "estimated", SLATE, italic=True)

    # ---- sensor constants ------------------------------------------------
    chip(ax, C1, Y_LID + 4.4, 13.2, 2.5,
         ["calib_velo_to_cam", "P_rect_02 · R_rect_00"])
    block_arrow(ax, C1, Y_LID + 3.15, C1, Y_LID + GH * 0.43 + 0.15,
                shaft=0.30, head_w=1.0, head_l=0.8, colour=VIOLET[0])

    chip(ax, X_GEO, Y_MID - 5.6, 15.4, 2.5,
         ["OXTS RT3003, per frame", "lat/lon · roll/pitch/yaw"])
    block_arrow(ax, X_GEO, Y_MID - 4.35, X_GEO, Y_MID - GH / 2 - 0.15,
                shaft=0.30, head_w=1.0, head_l=0.8, colour=VIOLET[0])

    # ---- arrows along the lanes ------------------------------------------
    for y, x_thumb in ((Y_CAM, 17.6), (Y_LID, 15.4)):
        block_arrow(ax, x_thumb, y, C1 - GW * 0.42, y)
        block_arrow(ax, C1 + GW * 0.42, y, C2 - GW * 0.42, y)
        block_arrow(ax, C2 + GW * 0.42, y, C3 - GW * 0.40, y)
    block_arrow(ax, C2 + GW * 0.40, Y_FALL, C3 - GW * 0.38, Y_FALL,
                dashed=True, colour=SLATE[0], head_w=1.7, head_l=1.3)

    # the fallback branches off the detections, and only when there is no LiDAR
    block_arrow(ax, C2, Y_CAM - GH * 0.46, C2, Y_FALL + GH * 0.44,
                dashed=True, colour=SLATE[0], head_w=1.7, head_l=1.3)
    ax.text(C2 + 1.1, (Y_CAM + Y_FALL) / 2, "no LiDAR", ha="left", va="center",
            fontsize=F_CHIP, style="italic", color="#5d6873")

    # ---- the merge -------------------------------------------------------
    ax.plot([X_BRACKET, X_BRACKET], [Y_LID, Y_CAM], color=INK, lw=3.4,
            solid_capstyle="butt", zorder=7)
    for y, x_from in ((Y_CAM, C3 + GW * 0.40), (Y_LID, C3 + GW * 0.36),
                      (Y_FALL, C3 + GW * 0.36)):
        ax.plot([x_from, X_BRACKET], [y, y], color=INK, lw=3.4,
                solid_capstyle="butt", zorder=7)
    block_arrow(ax, X_BRACKET, Y_MID, X_REC - GW * 0.46, Y_MID)

    # ---- the shared representation ---------------------------------------
    iso_box(ax, X_REC, Y_MID, GW * 0.78, GH * 0.80, 1.5,
            [(SLATE, True), (BLUE, False), (AMBER, False)], nx=4)
    label(ax, X_REC, Y_MID + GH / 2 + 0.2, "Object Record",
          "class + position", above=True)

    block_arrow(ax, X_REC + GW * 0.46, Y_MID, X_GEO - GW * 0.42, Y_MID)
    funnel(ax, X_GEO, Y_MID, GW * 0.74, GH, TEAL)
    label(ax, X_GEO, Y_MID + GH / 2 + 0.2, "Geo-referencing",
          "ENU → WGS84", above=True)

    block_arrow(ax, X_GEO + GW * 0.42, Y_MID, X_WORLD - GW * 0.44, Y_MID)
    iso_box(ax, X_WORLD, Y_MID, GW * 0.70, GH * 0.62, 1.5, [(TEAL, False)], nx=3)
    label(ax, X_WORLD, Y_MID + GH / 2 + 0.2, "World Objects",
          "lat/lon · size · bearing", above=True)

    # ---- fan out to the views --------------------------------------------
    y_cards = [Y_MID + 8.7, Y_MID, Y_MID - 8.7]
    ax.plot([X_WORLD + GW * 0.44, X_FAN], [Y_MID, Y_MID], color=INK, lw=3.4,
            solid_capstyle="butt", zorder=7)
    ax.plot([X_FAN, X_FAN], [y_cards[-1], y_cards[0]], color=INK, lw=3.4,
            solid_capstyle="butt", zorder=7)
    for y in y_cards:
        block_arrow(ax, X_FAN, y, X_CARD - CARD_W / 2 - 0.35, y)

    ax.text(X_CARD, H - 0.5, "Synchronised Views", ha="center", va="top",
            fontsize=F_HEAD)

    cards = [
        (data["map"], "Map · WGS84"),
        (data["scene3d"], "3D Scene · ENU"),
        (data["assistant"], "Scene Graph + Assistant"),
    ]
    for (image, caption), y in zip(cards, y_cards):
        result_card(ax, X_CARD, y, CARD_W, image, caption,
                    CARD_IMG_H, CARD_CAP_H)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pipeline.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  pipeline.png  ({path.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    data = collect()
    aspect = CARD_W / CARD_IMG_H

    data["map"] = map_thumb(data["placed"], data["track"], aspect=aspect)
    data["assistant"] = assistant_thumb(
        data["graph"], width=int(250 * aspect), height=250)

    scene3d = OUT / "scene-3d.png"
    if not scene3d.exists():
        print(f"Missing {scene3d}. Capture it from the running app first "
              f"(Export -> Screenshot with the 3D pane in front).",
              file=sys.stderr)
        raise SystemExit(1)
    data["scene3d"] = crop_to_aspect(Image.open(scene3d).convert("RGB"), aspect)

    draw(data)
    print(f"\nWrote the architecture figure to "
          f"{(OUT / 'pipeline.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
