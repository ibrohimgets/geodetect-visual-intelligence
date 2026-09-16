"""
Geo Intelligence Assistant: natural language over the detection results.

The design principle here is that the model is given **measurements, not
pixels**. It never sees the image. It receives the structured scene graph --
classes, confidences, distances, bearings, lat/lon, spatial relationships --
and its job is to translate a question into operations over that data. So
"how far is the bus" is answered from a number the geometry produced, not from
a model looking at a photo and estimating.

The second half of the design is that the assistant can *act*. Its tools do not
just return answers, they drive the viewer: selecting objects highlights them
simultaneously on the image, the map, the 3D scene and the table. Asking
"show me all people within 25 metres of the bus" runs a real spatial query and
the result lights up across every view.

The API key is read from the OPENAI_API_KEY environment variable. It is never
stored in this repository.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from . import scene_graph as sg
from .categories import CATEGORIES

log = logging.getLogger("cadsoftware.llm")

DEFAULT_MODEL = os.environ.get("GEODETECT_LLM_MODEL", "gpt-5.4-mini")

# Guard rails on how much scene goes into a prompt. A frame with 200 detections
# would otherwise produce a very large request for very little extra insight.
MAX_OBJECTS_IN_CONTEXT = 120
MAX_HISTORY_TURNS = 8


SYSTEM_PROMPT = """You are the Geo Intelligence Assistant inside GeoDetect, a \
visual-intelligence tool.

An image or video frame has been processed by a YOLO object detector, and each \
detection has been projected onto the ground plane to estimate its real-world \
position. You are given the resulting scene graph as structured data. You never \
see the image itself -- you reason over the measurements.

WHAT THE DATA MEANS
- distance_m is the horizontal ground distance from the camera, in metres.
- side is left / ahead / right from the camera's point of view.
- relative_bearing_deg is negative to the left of the camera axis, positive to \
the right. compass_bearing_deg is a true bearing, 0 = north.
- latitude/longitude are WGS84. east_m/north_m are metres from the camera.
- est_size_m is [width, height] estimated from the projection geometry.
- located: false means the object could not be placed on the ground (it was \
above the horizon or beyond max range). Such objects have no position at all -- \
say so rather than guessing.

WHERE EACH POSITION CAME FROM
Every located object carries `position_source`, and the two are genuinely \
different in quality. Read it before saying anything about how a number was \
obtained -- do not assume.

- position_source "lidar": MEASURED. Real LiDAR returns landed inside that \
object's box and gave its true range. `lidar_points` is how many, and `depth_m` \
is the measured range along the optical axis. Call these measured, not \
estimated. They are still subject to the usual sensor limits, but they are not \
a guess.
- position_source "ground_plane": ESTIMATED. No usable LiDAR return, so the \
position comes from projecting the bottom of the box onto an assumed flat \
ground plane using a single camera. Call these estimated, and say so when a \
question turns on precision.

The scene summary reports the split as `lidar_measured_count` and \
`ground_plane_estimated_count`. A scene can contain both. Never describe a \
measured position as an estimate, or an estimate as a measurement, and never \
invent a number that is not in the data.

HOW TO ANSWER
- Be concise and specific. Quote real numbers from the data.
- Refer to objects by their label (for example Person #1 or car_02).
- When the user asks to see, show, highlight, find or filter objects, CALL THE \
select_objects TOOL so the viewer actually highlights them, then briefly say \
what you selected.
- For questions that are purely informational ("how many people are there"), \
just answer from the data; no tool call is needed.
- If nothing matches, say so plainly.
"""


# --------------------------------------------------------------------------
# Tools -- what the assistant is allowed to do to the viewer
# --------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "select_objects",
            "description": (
                "Run a spatial query over the detected objects and highlight the "
                "matches across the image, map, 3D scene and table. All filters "
                "combine with AND. Use this whenever the user wants to see, show, "
                "highlight, filter or find objects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array", "items": {"type": "integer"},
                        "description": "Specific object ids, when you already know which ones.",
                    },
                    "classes": {
                        "type": "array", "items": {"type": "string"},
                        "description": "COCO class names, e.g. ['person','car'].",
                    },
                    "categories": {
                        "type": "array", "items": {"type": "string"},
                        "description": (
                            "Category groups. One or more of: "
                            + ", ".join(CATEGORIES.keys())
                        ),
                    },
                    "max_distance_m": {
                        "type": "number",
                        "description": "Keep objects no further than this from the camera.",
                    },
                    "min_distance_m": {
                        "type": "number",
                        "description": "Keep objects at least this far from the camera.",
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Keep detections at or above this confidence (0-1).",
                    },
                    "near_object_id": {
                        "type": "integer",
                        "description": (
                            "Keep only objects near THIS object. Use together with "
                            "near_radius_m, e.g. people within 25 m of the bus."
                        ),
                    },
                    "near_radius_m": {
                        "type": "number",
                        "description": "Radius in metres for near_object_id.",
                    },
                    "side": {
                        "type": "string", "enum": ["left", "ahead", "right"],
                        "description": "Keep objects on this side of the camera axis.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short human-readable description of this selection.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_selection",
            "description": "Remove any highlight and show all detected objects again.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focus_object",
            "description": (
                "Centre the map and the 3D scene on one object and select it. "
                "Use when the user asks where a single specific object is."
            ),
            "parameters": {
                "type": "object",
                "properties": {"object_id": {"type": "integer"}},
                "required": ["object_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_view",
            "description": "Bring a particular view to the front.",
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "enum": ["image", "map", "scene", "grid"]},
                },
                "required": ["view"],
            },
        },
    },
]


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------


def api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key or None


def is_available() -> bool:
    if not api_key():
        return False
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def status() -> dict[str, Any]:
    """What the UI shows in the assistant header."""
    if not api_key():
        return {"available": False, "model": DEFAULT_MODEL,
                "reason": "OPENAI_API_KEY is not set in the environment"}
    try:
        import openai  # noqa: F401
    except ImportError:
        return {"available": False, "model": DEFAULT_MODEL,
                "reason": "the openai package is not installed (pip install openai)"}
    return {"available": True, "model": DEFAULT_MODEL, "reason": ""}


# --------------------------------------------------------------------------
# Tool execution
# --------------------------------------------------------------------------


def _run_tool(name: str, args: dict, graph: sg.SceneGraph) -> tuple[dict, Optional[dict]]:
    """Execute one tool call.

    Returns (result_for_the_model, action_for_the_frontend). The model gets a
    factual summary so it can phrase a natural answer; the frontend gets an
    instruction so the views actually change.
    """
    if name == "select_objects":
        matches = sg.select(
            graph,
            ids=args.get("ids"),
            classes=args.get("classes"),
            categories=args.get("categories"),
            min_confidence=args.get("min_confidence"),
            max_distance_m=args.get("max_distance_m"),
            min_distance_m=args.get("min_distance_m"),
            near_object_id=args.get("near_object_id"),
            near_radius_m=args.get("near_radius_m"),
            side=args.get("side"),
        )
        matches = sorted(matches, key=lambda n: (not n.located, n.distance_m))
        result = {
            "matched": len(matches),
            "objects": [
                {
                    "id": n.id, "label": n.label, "class": n.class_name,
                    "distance_m": round(n.distance_m, 1) if n.located else None,
                    "side": n.side if n.located else None,
                    "confidence": round(n.confidence, 2),
                }
                for n in matches
            ],
        }
        action = {
            "type": "select",
            "ids": [n.id for n in matches],
            "label": args.get("reason") or _describe_filter(args),
        }
        return result, action

    if name == "clear_selection":
        return {"cleared": True}, {"type": "clear"}

    if name == "focus_object":
        node = graph.by_id(int(args.get("object_id", -1)))
        if node is None:
            return {"error": "no object with that id"}, None
        return (
            {
                "id": node.id, "label": node.label, "class": node.class_name,
                "located": node.located,
                "distance_m": round(node.distance_m, 1) if node.located else None,
                "side": node.side if node.located else None,
                "latitude": round(node.lat, 6) if node.located else None,
                "longitude": round(node.lon, 6) if node.located else None,
            },
            {"type": "focus", "id": node.id, "label": node.label},
        )

    if name == "switch_view":
        view = args.get("view", "grid")
        return {"view": view}, {"type": "view", "view": view}

    return {"error": f"unknown tool {name}"}, None


def _describe_filter(args: dict) -> str:
    """A short label for the selection chip in the UI."""
    bits: list[str] = []
    if args.get("classes"):
        bits.append(", ".join(args["classes"]))
    if args.get("categories"):
        bits.append(", ".join(args["categories"]))
    if not bits:
        bits.append("objects")
    if args.get("max_distance_m") is not None:
        bits.append(f"within {args['max_distance_m']:g} m")
    if args.get("min_distance_m") is not None:
        bits.append(f"beyond {args['min_distance_m']:g} m")
    if args.get("side"):
        bits.append(f"on the {args['side']}")
    if args.get("near_object_id") is not None:
        radius = args.get("near_radius_m")
        bits.append(f"near #{args['near_object_id']}"
                    + (f" ({radius:g} m)" if radius else ""))
    return " · ".join(bits)


# --------------------------------------------------------------------------
# The main entry point
# --------------------------------------------------------------------------


def ask(
    question: str,
    graph: sg.SceneGraph,
    history: Optional[list[dict]] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Answer a question about the scene, possibly acting on the viewer.

    Returns a dict with the answer text, any actions the frontend should apply,
    and a transcript of the tool calls so the UI can show its working.
    """
    if not is_available():
        return _fallback(question, graph, status()["reason"])

    from openai import OpenAI

    model = model or DEFAULT_MODEL
    client = OpenAI()

    context = graph.to_context()
    # Trim very large scenes; the summary still reports the true total.
    if len(context["objects"]) > MAX_OBJECTS_IN_CONTEXT:
        context["objects"] = context["objects"][:MAX_OBJECTS_IN_CONTEXT]
        context["truncated_to"] = MAX_OBJECTS_IN_CONTEXT

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "Current scene:\n" + json.dumps(context, separators=(",", ":")),
        },
    ]
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)[:2000]})
    messages.append({"role": "user", "content": question})

    actions: list[dict] = []
    trace: list[dict] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    answer = ""

    # Up to three rounds: the model calls a tool, sees the result, and may
    # refine once before answering. More than that is almost always a loop.
    for _ in range(3):
        try:
            response = _create(client, model=model, messages=messages, tools=TOOLS)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            log.exception("LLM request failed")
            return {
                "answer": f"The assistant could not reach the language model: {exc}",
                "actions": [], "tool_calls": [], "model": model,
                "error": str(exc), "fallback": False,
            }

        if response.usage:
            usage_total["prompt_tokens"] += response.usage.prompt_tokens or 0
            usage_total["completion_tokens"] += response.usage.completion_tokens or 0

        message = response.choices[0].message
        answer = message.content or answer

        if not message.tool_calls:
            break

        # Echo the assistant's tool-call message back before the results, which
        # the chat completions API requires.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })

        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            result, action = _run_tool(call.function.name, args, graph)
            if action:
                actions.append(action)
            trace.append({
                "tool": call.function.name,
                "arguments": args,
                "matched": result.get("matched"),
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, separators=(",", ":"))[:6000],
            })

    return {
        "answer": answer or "Done.",
        "actions": actions,
        "tool_calls": trace,
        "model": model,
        "usage": usage_total,
        "fallback": False,
    }


def _create(client, *, model: str, messages: list, tools: list):
    """Call the chat API, coping with differences between model generations.

    Newer models reject `max_tokens` and want `max_completion_tokens`; some
    reasoning models also reject a non-default `temperature`. Rather than keep a
    table of which model wants what, we try the modern form and step back on the
    specific complaint the API makes.
    """
    attempts: list[dict[str, Any]] = [
        {"max_completion_tokens": 900, "temperature": 0},
        {"max_completion_tokens": 900},
        {"max_tokens": 900, "temperature": 0},
        {},
    ]
    last: Exception | None = None
    for extra in attempts:
        try:
            return client.chat.completions.create(
                model=model, messages=messages, tools=tools, **extra
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            # Only step back for a parameter complaint; anything else (auth,
            # rate limit, network) should surface immediately.
            if "unsupported" in message or "unknown parameter" in message \
                    or "does not support" in message:
                last = exc
                continue
            raise
    raise last if last else RuntimeError("chat completion failed")


# --------------------------------------------------------------------------
# Fallback: keep the panel useful when there is no API key
# --------------------------------------------------------------------------


def _fallback(question: str, graph: sg.SceneGraph, reason: str) -> dict[str, Any]:
    """A tiny rule-based responder for when the language model is unavailable.

    It covers the handful of questions that are pure lookups so the tool still
    demonstrates the pipeline on a machine with no API key. It says clearly that
    it is not the language model, because pretending otherwise would be worse
    than being limited.
    """
    q = question.lower()
    located = graph.located
    note = f"  \n\n*(Assistant running without the language model: {reason}.)*"

    # "how many X"
    count_match = re.search(r"how many (\w+)", q)
    if count_match:
        word = count_match.group(1).rstrip("s")
        matches = [n for n in graph.nodes if word in n.class_name.lower()]
        return _fb(f"{len(matches)} {word}{'' if len(matches) == 1 else 's'} detected." + note,
                   [{"type": "select", "ids": [n.id for n in matches], "label": word}]
                   if matches else [])

    if "closest" in q or "nearest" in q:
        if not located:
            return _fb("No object has an estimated position." + note, [])
        n = min(located, key=lambda x: x.distance_m)
        return _fb(f"{n.label} is closest, about {n.distance_m:.1f} m away (estimated)." + note,
                   [{"type": "focus", "id": n.id, "label": n.label}])

    if "farthest" in q or "furthest" in q:
        if not located:
            return _fb("No object has an estimated position." + note, [])
        n = max(located, key=lambda x: x.distance_m)
        return _fb(f"{n.label} is farthest, about {n.distance_m:.1f} m away (estimated)." + note,
                   [{"type": "focus", "id": n.id, "label": n.label}])

    for category in CATEGORIES:
        if category.rstrip("s") in q:
            matches = sg.select(graph, categories=[category])
            return _fb(f"{len(matches)} object(s) in '{category}'." + note,
                       [{"type": "select", "ids": [n.id for n in matches],
                         "label": category}])

    if "summar" in q or "describe" in q or "scene" in q:
        counts: dict[str, int] = {}
        for n in graph.nodes:
            counts[n.class_name] = counts.get(n.class_name, 0) + 1
        parts = ", ".join(f"{v} {k}{'' if v == 1 else 's'}" for k, v in counts.items())
        extra = ""
        if located:
            nearest = min(located, key=lambda x: x.distance_m)
            extra = f" Nearest is {nearest.label} at about {nearest.distance_m:.1f} m."
        return _fb(f"{len(graph.nodes)} objects: {parts or 'none'}.{extra}" + note, [])

    return _fb(
        "The language model is not configured, so I can only handle simple "
        "lookups: object counts, nearest and farthest, category filters and a "
        "scene summary." + note,
        [],
    )


def _fb(answer: str, actions: list[dict]) -> dict[str, Any]:
    return {"answer": answer, "actions": actions, "tool_calls": [],
            "model": "rule-based fallback", "usage": {}, "fallback": True}
