#!/usr/bin/env python
"""
Generate the bundled demo clip for video mode.

    python tools/make_sample_video.py

There is no video in this repository and nothing is downloaded, so the sample is
synthesised locally: a virtual camera pans and slowly zooms across one of the
bundled street photographs, writing the moving window out as an MP4.

The content is a real photograph, so the detector sees genuine people and a real
bus, and because the window moves they travel across the frame -- which is
exactly what the tracker and the motion trails need in order to be worth
anything. It is a synthetic camera move over real imagery, and the sample is
labelled as such in the UI rather than passed off as drone footage.

Swap in your own MP4 any time; this exists so the feature is demonstrable out of
the box.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "samples" / "street-bus.jpg"

FPS = 25
DURATION_S = 8.0
OUT_W, OUT_H = 720, 720

# Codec choice matters more than it looks.
#
# The clip has to satisfy two different decoders: OpenCV on the server, which
# will read it back for tracking, and the *browser*, which plays it behind the
# detection overlay. OpenCV happily writes MPEG-4 Part 2 ("mp4v"), and no
# browser can play it -- the video element just sits at readyState 0.
#
# H.264 would be ideal but OpenCV needs the OpenH264 DLL for it, which is often
# missing on Windows. VP8 in a WebM container needs no extra library and every
# modern browser decodes it, so that is the first choice, with H.264 tried first
# in case this machine does have it.
CODECS = [
    ("avc1", ".mp4"),   # H.264, if this OpenCV build can encode it
    ("VP80", ".webm"),  # VP8, no extra dependency, universally playable
    ("VP90", ".webm"),  # VP9
]

# FOURCC tags a browser can decode, as they appear when read back from a file.
BROWSER_SAFE = {"h264", "avc1", "vp80", "vp90", "vp08", "vp09"}


def _fourcc_of(path: Path) -> str:
    """The codec tag actually stored in a written file, lowercased."""
    if not path.exists() or path.stat().st_size == 0:
        return ""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return ""
    raw = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    return "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)).strip().lower()


def main() -> int:
    if not SOURCE.exists():
        print(f"Source image not found: {SOURCE}", file=sys.stderr)
        return 1

    image = cv2.imread(str(SOURCE))
    if image is None:
        print(f"Could not read {SOURCE}", file=sys.stderr)
        return 1

    # Work at 2x so the moving crop stays sharp after resizing to the output.
    image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    src_h, src_w = image.shape[:2]

    frames = int(FPS * DURATION_S)

    # Pick a codec by probing, not by trusting isOpened().
    #
    # VideoWriter.isOpened() returns True even when the encoder it wanted is
    # missing and it quietly fell back to something else -- which is how the
    # first version of this script produced an MPEG-4 Part 2 file that no
    # browser would play. So write a throwaway clip with each candidate, read
    # the FOURCC back out of the finished file, and only accept a tag the
    # browser can actually decode.
    chosen = None
    for fourcc, ext in CODECS:
        probe = ROOT / "data" / "samples" / f".probe{ext}"
        w = cv2.VideoWriter(str(probe), cv2.VideoWriter_fourcc(*fourcc), FPS, (OUT_W, OUT_H))
        if w.isOpened():
            blank = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
            for _ in range(3):
                w.write(blank)
        w.release()

        tag = _fourcc_of(probe)
        probe.unlink(missing_ok=True)
        if tag in BROWSER_SAFE:
            chosen, extension = fourcc, ext
            break

    if chosen is None:
        print("No browser-playable codec available from OpenCV on this machine.\n"
              "Supply your own MP4 instead - any H.264 file works.", file=sys.stderr)
        return 1

    output = ROOT / "data" / "samples" / f"street-pan{extension}"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*chosen), FPS,
                             (OUT_W, OUT_H))
    if not writer.isOpened():
        print("Could not open the video writer", file=sys.stderr)
        return 1

    # Remove any stale clip written by an earlier run with another codec.
    for _, ext in CODECS:
        stale = ROOT / "data" / "samples" / f"street-pan{ext}"
        if stale != output and stale.exists():
            stale.unlink()

    for i in range(frames):
        t = i / (frames - 1)  # 0 -> 1

        # Ease in and out so the move looks like a camera rather than a slide.
        eased = 0.5 - 0.5 * math.cos(math.pi * t)

        # Gentle zoom out across the clip, and a diagonal pan.
        scale = 0.55 + 0.25 * eased
        crop_w = int(src_w * scale)
        crop_h = int(crop_w * OUT_H / OUT_W)
        crop_h = min(crop_h, src_h)
        crop_w = min(crop_w, src_w)

        max_x = max(0, src_w - crop_w)
        max_y = max(0, src_h - crop_h)
        x = int(max_x * (0.15 + 0.7 * eased))
        y = int(max_y * (0.75 - 0.55 * eased))

        crop = image[y:y + crop_h, x:x + crop_w]
        writer.write(cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA))

    writer.release()

    # Read it back: a writer that opened is not proof of a file that decodes.
    tag = _fourcc_of(output)
    check = cv2.VideoCapture(str(output))
    ok, _ = check.read() if check.isOpened() else (False, None)
    decoded = int(check.get(cv2.CAP_PROP_FRAME_COUNT)) if check.isOpened() else 0
    check.release()
    if not ok:
        print(f"Wrote {output.name} but it could not be read back", file=sys.stderr)
        return 1

    size_kb = output.stat().st_size / 1024
    print(f"Wrote {output.relative_to(ROOT)}  "
          f"(codec {chosen} -> {tag}, {decoded} frames, {DURATION_S:.0f}s, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
