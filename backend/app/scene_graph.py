"""
Scene graph: turning coordinates into relationships.

Geo-referencing gives every detection a position. That is necessary but not
sufficient for answering the questions people actually ask, which are
relational: *which* object is closest, is anyone standing near the bus, what is
to the left of the car. This module computes that layer.

    object -> position -> distance -> spatial relationships

Everything here is derived from the geo stage's ENU coordinates, so it inherits
the same flat-ground assumption and the same honesty about it: relationships
between objects that have no ground fix are simply not asserted.

Frames of reference
-------------------
"Left" and "right" are from the *camera's* point of view, which is what a person
looking at the image means. We get that by taking each object's compass bearing
from the camera and subtracting the camera's own heading, giving a relative
bearing where negative is left of the optical axis and positive is right.

"Near" and "far" are scene-relative. Two metres apart is close on a street and
almost touching from a drone at 80 m, so a fixed threshold would be wrong at one
scale or the other. The radius scales with the median object distance, clamped
to a sensible band -- see `proximity_radius`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable, Optional

from .categories import category_of

# Relative bearings inside this cone count as "ahead" rather than left or right.
_CENTRE_CONE_DEG = 8.0

# Two objects must differ by at least this much in relative bearing before we
# are willing to say one is left of the other.
_SIDE_SEPARATION_DEG = 6.0

# Bounds on the adaptive proximity radius, in metres.
_MIN_NEAR_RADIUS_M = 1.5
_MAX_NEAR_RADIUS_M = 30.0


@dataclass
class Relation:
    """One directed relationship, e.g. 'left of Bus #1'."""

    kind: str            # left_of | right_of | near | in_front_of | behind
    target_id: int
    target_label: str
    distance_m: float = 0.0

    def phrase(self) -> str:
        words = {
            "left_of": "left of",
            "right_of": "right of",
            "near": "near",
            "in_front_of": "in front of",
            "behind": "behind",
        }[self.kind]
        if self.kind == "near":
            return f"{words} {self.target_label} ({self.distance_m:.1f} m apart)"
        return f"{words} {self.target_label}"


@dataclass
class SceneNode:
    """One object, with everything we know about where it is and what is around it."""

    id: int
    label: str
    class_name: str
    category: str
    confidence: float

    # Position, all derived from the geo stage.
    distance_m: float = 0.0
    bearing_deg: float = 0.0
    relative_bearing_deg: float = 0.0
    east_m: float = 0.0
    north_m: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0

    side: str = "ahead"          # left | ahead | right
    distance_band: str = "mid"   # near | mid | far
    group_id: Optional[int] = None
    located: bool = True         # False when the object has no ground fix

    # How the position was obtained. The assistant must never describe a
    # LiDAR-measured range as an estimate, or the other way round, so the
    # provenance travels with the measurement rather than being inferred from
    # context.
    source: str = "ground_plane"  # lidar | ground_plane
    lidar_points: int = 0
    depth_m: float = 0.0

    relations: list[Relation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The compact form handed to the language model.

        Deliberately flat and small: the model reasons better over a tidy table
        than over deeply nested JSON, and it keeps the token cost down.
        """
        out: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "class": self.class_name,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "located": self.located,
        }
        if self.located:
            out.update({
                "distance_m": round(self.distance_m, 1),
                "side": self.side,
                "relative_bearing_deg": round(self.relative_bearing_deg, 1),
                "compass_bearing_deg": round(self.bearing_deg, 1),
                "latitude": round(self.lat, 6),
                "longitude": round(self.lon, 6),
                "east_m": round(self.east_m, 1),
                "north_m": round(self.north_m, 1),
                "est_size_m": [round(self.width_m, 1), round(self.height_m, 1)],
                "distance_band": self.distance_band,
                "position_source": self.source,
            })
            if self.source == "lidar":
                out["lidar_points"] = self.lidar_points
                out["depth_m"] = round(self.depth_m, 1)
            if self.group_id is not None:
                out["group"] = self.group_id
            if self.relations:
                out["relations"] = [
                    {"kind": r.kind, "target": r.target_label, "target_id": r.target_id}
                    for r in self.relations
                ]
        return out


@dataclass
class SceneGraph:
    nodes: list[SceneNode]
    groups: list[list[int]]
    near_radius_m: float
    camera: dict[str, Any]

    # ------------------------------------------------------------------ lookup

    def by_id(self, object_id: int) -> Optional[SceneNode]:
        return next((n for n in self.nodes if n.id == object_id), None)

    @property
    def located(self) -> list[SceneNode]:
        return [n for n in self.nodes if n.located]

    # ------------------------------------------------------------------ output

    def to_context(self) -> dict[str, Any]:
        """The structured payload the assistant reasons over.

        This is the whole point of the design: the model is handed measured
        numbers, not an image to guess from.
        """
        located = self.located
        counts: dict[str, int] = {}
        for n in self.nodes:
            counts[n.class_name] = counts.get(n.class_name, 0) + 1

        measured = sum(1 for n in located if n.source == "lidar")
        summary: dict[str, Any] = {
            "object_count": len(self.nodes),
            "located_count": len(located),
            "lidar_measured_count": measured,
            "ground_plane_estimated_count": len(located) - measured,
            "class_counts": counts,
            "near_radius_m": round(self.near_radius_m, 1),
            "camera": self.camera,
        }
        if located:
            nearest = min(located, key=lambda n: n.distance_m)
            farthest = max(located, key=lambda n: n.distance_m)
            summary["nearest"] = {"id": nearest.id, "label": nearest.label,
                                  "distance_m": round(nearest.distance_m, 1)}
            summary["farthest"] = {"id": farthest.id, "label": farthest.label,
                                   "distance_m": round(farthest.distance_m, 1)}

        return {
            "summary": summary,
            "objects": [n.to_dict() for n in self.nodes],
            "groups": [
                {"id": i, "members": members}
                for i, members in enumerate(self.groups) if len(members) > 1
            ],
        }

    def to_tree(self, limit: int = 40) -> str:
        """The scene as an indented tree, for display in the UI.

            Person #1
            ├─ 18.2 m from camera
            ├─ left of Bus #1
            └─ near Car #2
        """
        lines: list[str] = []
        for node in self.nodes[:limit]:
            lines.append(node.label)

            branches: list[str] = []
            if node.located:
                how = (f"LiDAR, {node.lidar_points} returns"
                       if node.source == "lidar" else "estimated")
                branches.append(f"{node.distance_m:.1f} m from camera ({how})")
                if node.side != "ahead":
                    branches.append(f"{node.side} of centre "
                                    f"({abs(node.relative_bearing_deg):.0f}°)")
                branches.extend(r.phrase() for r in node.relations)
            else:
                branches.append("no ground fix (above horizon or out of range)")

            for i, text in enumerate(branches):
                connector = "└─" if i == len(branches) - 1 else "├─"
                lines.append(f"{connector} {text}")
            lines.append("")

        if len(self.nodes) > limit:
            lines.append(f"... and {len(self.nodes) - limit} more")
        return "\n".join(lines).rstrip()


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def proximity_radius(distances: list[float]) -> float:
    """How close two objects must be before we call them 'near' each other.

    Scaled to the scene, because proximity is relative: a metre apart is
    touching on a drone frame and a whole car-length on a desk. We take a
    fraction of the median object distance and clamp it so neither extreme
    produces something silly.
    """
    if not distances:
        return 5.0
    return max(_MIN_NEAR_RADIUS_M, min(_MAX_NEAR_RADIUS_M, 0.18 * median(distances)))


def _label_for(class_name: str, ordinal: int, track_id: Optional[int]) -> str:
    """'person_01' style when tracking, 'Person #1' style for a still image."""
    if track_id is not None:
        return f"{class_name.replace(' ', '_')}_{track_id:02d}"
    return f"{class_name.title()} #{ordinal}"


def build(detections: Iterable[dict], camera: dict) -> SceneGraph:
    """Build the scene graph from the geo stage's output.

    `detections` are the serialised DetectionOut dicts, and `camera` the
    serialised CameraOut. Both come straight from the existing pipeline, so the
    scene graph is a pure function of what the earlier stages already produced.
    """
    detections = list(detections)
    heading = float(camera.get("heading_deg", 0.0))

    nodes: list[SceneNode] = []
    per_class_count: dict[str, int] = {}

    for det in detections:
        world = det.get("world", {})
        class_name = det["class_name"]
        per_class_count[class_name] = per_class_count.get(class_name, 0) + 1
        track_id = det.get("track_id")

        node = SceneNode(
            id=det["id"],
            label=_label_for(class_name, per_class_count[class_name], track_id),
            class_name=class_name,
            category=category_of(class_name),
            confidence=float(det.get("confidence", 0.0)),
            located=bool(world.get("valid")),
        )

        if node.located:
            node.distance_m = float(world.get("ground_range_m", 0.0))
            node.bearing_deg = float(world.get("bearing_deg", 0.0))
            node.east_m = float(world.get("east_m", 0.0))
            node.north_m = float(world.get("north_m", 0.0))
            node.lat = float(world.get("lat", 0.0))
            node.lon = float(world.get("lon", 0.0))
            node.width_m = float(world.get("est_width_m", 0.0))
            node.height_m = float(world.get("est_height_m", 0.0))
            node.source = str(world.get("source", "ground_plane"))
            node.lidar_points = int(world.get("lidar_points", 0) or 0)
            node.depth_m = float(world.get("depth_m", 0.0) or 0.0)

            # Relative bearing: where the object sits compared with where the
            # camera is pointing. Wrapped into [-180, 180] so the sign is the
            # answer -- negative is left, positive is right.
            rel = (node.bearing_deg - heading + 540.0) % 360.0 - 180.0
            node.relative_bearing_deg = rel
            if rel < -_CENTRE_CONE_DEG:
                node.side = "left"
            elif rel > _CENTRE_CONE_DEG:
                node.side = "right"
            else:
                node.side = "ahead"

        nodes.append(node)

    located = [n for n in nodes if n.located]
    distances = [n.distance_m for n in located]
    near_radius = proximity_radius(distances)

    _assign_distance_bands(located)
    groups = _cluster(located, near_radius * 1.6)
    _assign_relations(located, near_radius)

    return SceneGraph(
        nodes=nodes,
        groups=groups,
        near_radius_m=near_radius,
        camera={
            "latitude": round(float(camera.get("lat", 0.0)), 6),
            "longitude": round(float(camera.get("lon", 0.0)), 6),
            "altitude_m": round(float(camera.get("altitude_m", 0.0)), 1),
            "heading_deg": round(heading, 1),
            "pitch_deg": round(float(camera.get("pitch_deg", 0.0)), 1),
            "hfov_deg": round(float(camera.get("hfov_deg", 0.0)), 1),
        },
    )


def _assign_distance_bands(located: list[SceneNode]) -> None:
    """Split objects into near / mid / far thirds of this scene's spread.

    Terciles rather than fixed metre thresholds, so "far" always means far
    *for this scene* instead of far in absolute terms.
    """
    if len(located) < 3:
        for n in located:
            n.distance_band = "near" if len(located) < 2 else "mid"
        return

    ordered = sorted(located, key=lambda n: n.distance_m)
    third = len(ordered) / 3.0
    for i, node in enumerate(ordered):
        node.distance_band = "near" if i < third else ("mid" if i < 2 * third else "far")


def _cluster(located: list[SceneNode], threshold: float) -> list[list[int]]:
    """Single-linkage clustering on ground positions, to find groups.

    Deliberately the simplest thing that works: objects join a group if they are
    within `threshold` of any current member. With the handful of detections a
    frame produces, the quadratic cost is irrelevant and the behaviour is easy
    to explain, which matters more here than asymptotics.
    """
    groups: list[list[SceneNode]] = []

    for node in located:
        joined = None
        for group in groups:
            if any(_ground_distance(node, other) <= threshold for other in group):
                group.append(node)
                joined = group
                break
        if joined is None:
            groups.append([node])

    # Merge groups that ended up linked through a later object.
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(_ground_distance(a, b) <= threshold
                       for a in groups[i] for b in groups[j]):
                    groups[i].extend(groups[j])
                    del groups[j]
                    merged = True
                    break
            if merged:
                break

    for index, group in enumerate(groups):
        for node in group:
            node.group_id = index

    return [[n.id for n in group] for group in groups]


def _ground_distance(a: SceneNode, b: SceneNode) -> float:
    return math.hypot(a.east_m - b.east_m, a.north_m - b.north_m)


def _assign_relations(located: list[SceneNode], near_radius: float,
                      max_per_node: int = 3) -> None:
    """Attach a few of the most informative relationships to each object.

    Every object is related to every other one in some way, but a list of all of
    them is noise. We keep the closest neighbours, and describe each with the
    single most useful relation -- proximity if they are close, otherwise which
    side it is on, otherwise depth.
    """
    for node in located:
        others = sorted(
            (o for o in located if o.id != node.id),
            key=lambda o: _ground_distance(node, o),
        )

        for other in others[:max_per_node]:
            gap = _ground_distance(node, other)

            if gap <= near_radius:
                node.relations.append(
                    Relation("near", other.id, other.label, gap)
                )
                continue

            bearing_gap = node.relative_bearing_deg - other.relative_bearing_deg
            if abs(bearing_gap) >= _SIDE_SEPARATION_DEG:
                kind = "left_of" if bearing_gap < 0 else "right_of"
                node.relations.append(Relation(kind, other.id, other.label, gap))
                continue

            # Roughly the same direction from the camera, so distinguish by depth.
            kind = "in_front_of" if node.distance_m < other.distance_m else "behind"
            node.relations.append(Relation(kind, other.id, other.label, gap))


# --------------------------------------------------------------------------
# Queries -- the operations the assistant's tools are built on
# --------------------------------------------------------------------------


def select(
    graph: SceneGraph,
    ids: Optional[list[int]] = None,
    classes: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    min_confidence: Optional[float] = None,
    max_distance_m: Optional[float] = None,
    min_distance_m: Optional[float] = None,
    near_object_id: Optional[int] = None,
    near_radius_m: Optional[float] = None,
    side: Optional[str] = None,
) -> list[SceneNode]:
    """Filter the scene. Every argument is an AND, and None means 'do not care'.

    This is the operation natural language gets translated into. "People within
    25 m of the bus" becomes classes=['person'], near_object_id=<bus>,
    near_radius_m=25.
    """
    from .categories import resolve_categories  # local import avoids a cycle

    result = list(graph.nodes)

    if ids is not None:
        wanted = set(ids)
        result = [n for n in result if n.id in wanted]

    if classes:
        wanted_classes = {c.strip().lower() for c in classes}
        result = [n for n in result if n.class_name.lower() in wanted_classes]

    if categories:
        wanted_classes = {c.lower() for c in resolve_categories(categories)}
        result = [n for n in result if n.class_name.lower() in wanted_classes]

    if min_confidence is not None:
        result = [n for n in result if n.confidence >= min_confidence]

    # Distance filters only make sense for objects that have a position at all.
    if max_distance_m is not None:
        result = [n for n in result if n.located and n.distance_m <= max_distance_m]

    if min_distance_m is not None:
        result = [n for n in result if n.located and n.distance_m >= min_distance_m]

    if side:
        want = side.strip().lower()
        result = [n for n in result if n.located and n.side == want]

    if near_object_id is not None:
        anchor = graph.by_id(near_object_id)
        if anchor is None or not anchor.located:
            return []
        radius = near_radius_m if near_radius_m is not None else graph.near_radius_m
        result = [
            n for n in result
            if n.located and n.id != anchor.id
            and _ground_distance(n, anchor) <= radius
        ]

    return result
