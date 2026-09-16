"""
Pull camera pose out of image EXIF metadata.

Drone photos (DJI, Autel, Parrot and friends) record where the aircraft was and
how the gimbal was pointed at the moment of capture. Phone photos usually carry
GPS and focal length too. When any of that is present we can populate the camera
panel from the file instead of asking the operator to type it in, which is
exactly what a real photogrammetry ingest would do.

Everything here is best-effort. Missing or malformed tags are simply skipped --
a photo with no metadata still works, the operator just fills the fields in.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

from PIL import Image, ExifTags

log = logging.getLogger("cadsoftware.exif")

# Reverse lookup: human-readable tag name -> numeric EXIF id
_TAG_IDS = {name: num for num, name in ExifTags.TAGS.items()}
_GPS_IDS = {name: num for num, name in ExifTags.GPSTAGS.items()}


def _to_float(value: Any) -> Optional[float]:
    """EXIF numbers arrive as ints, floats or IFDRational. Normalise them."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_degrees(dms: Any, ref: Optional[str]) -> Optional[float]:
    """Convert EXIF degrees/minutes/seconds plus a N/S/E/W ref into signed degrees."""
    try:
        d, m, s = (float(x) for x in dms)
    except (TypeError, ValueError, ZeroDivisionError):
        return None

    deg = d + m / 60.0 + s / 3600.0
    if ref and str(ref).upper().strip() in ("S", "W"):
        deg = -deg
    return deg


def _xmp_float(blob: str, *keys: str) -> Optional[float]:
    """Find a numeric XMP attribute, e.g. drone-dji:GimbalPitchDegree="-45.0".

    DJI writes gimbal angles into an XMP packet rather than standard EXIF, so we
    scan the raw bytes for the attribute rather than parsing the whole XML.
    """
    for key in keys:
        match = re.search(rf'{re.escape(key)}\s*=\s*"([+-]?[\d.]+)"', blob)
        if not match:
            match = re.search(rf"{re.escape(key)}>\s*([+-]?[\d.]+)\s*<", blob)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def extract_camera_hints(path: str, image_width: int) -> dict:
    """Read whatever camera parameters the file is willing to tell us.

    Returns a dict with any of: lat, lon, altitude_m, heading_deg, pitch_deg,
    hfov_deg, plus a `found` list naming which fields came from metadata, and
    `make` / `model` for display. Keys are absent when the tag was missing.
    """
    hints: dict[str, Any] = {}
    found: list[str] = []

    try:
        with Image.open(path) as img:
            exif = img.getexif()
            xmp_blob = ""
            # DJI stores gimbal orientation in an XMP packet.
            for seg in (img.info.get("XML:com.adobe.xmp"), img.info.get("xmp")):
                if isinstance(seg, bytes):
                    xmp_blob += seg.decode("utf-8", errors="ignore")
                elif isinstance(seg, str):
                    xmp_blob += seg

            if not exif and not xmp_blob:
                return {"found": []}

            make = exif.get(_TAG_IDS.get("Make", -1))
            model = exif.get(_TAG_IDS.get("Model", -1))
            if make:
                hints["make"] = str(make).strip()
            if model:
                hints["model"] = str(model).strip()

            # ---------------------------------------------------------- GPS
            gps = {}
            try:
                gps_ifd = exif.get_ifd(_TAG_IDS.get("GPSInfo", -1))
                if gps_ifd:
                    gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
            except Exception:  # noqa: BLE001 - malformed GPS block, keep going
                gps = {}

            lat = _dms_to_degrees(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
            lon = _dms_to_degrees(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
            if lat is not None and lon is not None and abs(lat) <= 90 and abs(lon) <= 180:
                hints["lat"] = lat
                hints["lon"] = lon
                found.append("position")

            # GPSAltitude is height above sea level, and ref 1 means below it.
            alt = _to_float(gps.get("GPSAltitude"))
            if alt is not None:
                if str(gps.get("GPSAltitudeRef", 0)) in ("1", "b'\\x01'"):
                    alt = -alt
                hints["gps_altitude_m"] = alt

            # Height above the take-off point is what our ground plane wants.
            rel_alt = _xmp_float(xmp_blob, "drone-dji:RelativeAltitude", "RelativeAltitude")
            if rel_alt is not None and rel_alt > 0:
                hints["altitude_m"] = abs(rel_alt)
                found.append("altitude")
            elif alt is not None and alt > 0:
                hints["altitude_m"] = alt
                found.append("altitude (above sea level)")

            # ------------------------------------------------------- heading
            heading = _xmp_float(
                xmp_blob, "drone-dji:GimbalYawDegree", "drone-dji:FlightYawDegree"
            )
            if heading is None:
                heading = _to_float(gps.get("GPSImgDirection"))
            if heading is not None:
                hints["heading_deg"] = heading % 360.0
                found.append("heading")

            # --------------------------------------------------------- pitch
            # DJI reports gimbal pitch as 0 at the horizon and -90 straight
            # down. We use "degrees below horizontal", so flip the sign.
            gimbal_pitch = _xmp_float(xmp_blob, "drone-dji:GimbalPitchDegree")
            if gimbal_pitch is not None:
                hints["pitch_deg"] = max(0.0, min(90.0, -gimbal_pitch))
                found.append("gimbal pitch")

            # ----------------------------------------------------------- FOV
            # Prefer the 35mm-equivalent focal length, which already accounts for
            # the sensor size. A 35mm frame is 36mm wide, so
            #   hfov = 2 * atan(18 / focal_35mm)
            f35 = _to_float(exif.get(_TAG_IDS.get("FocalLengthIn35mmFilm", -1)))
            if f35 and f35 > 1:
                hints["hfov_deg"] = math.degrees(2.0 * math.atan(18.0 / f35))
                found.append("field of view")
                hints["focal_35mm"] = f35
            else:
                hints["focal_length_mm"] = _to_float(
                    exif.get(_TAG_IDS.get("FocalLength", -1))
                )

    except Exception as exc:  # noqa: BLE001 - metadata is a bonus, never fatal
        log.debug("EXIF extraction skipped for %s: %s", path, exc)
        return {"found": []}

    hints["found"] = found
    return hints
