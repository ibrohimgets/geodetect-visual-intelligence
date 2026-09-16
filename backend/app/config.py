"""Project paths and settings, all in one place."""

from __future__ import annotations

import os
from pathlib import Path

# backend/app/config.py -> backend/app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DIR = DATA_DIR / "samples"
MODELS_DIR = PROJECT_ROOT / "models"

# Real sensor data. Each KITTI raw drive lives under a date folder here,
# e.g. data/kitti/2011_09_26/2011_09_26_drive_0048_sync/
KITTI_DIR = DATA_DIR / "kitti"

for _d in (UPLOAD_DIR, SAMPLE_DIR, MODELS_DIR, KITTI_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Which pretrained checkpoint to run. The 'n' (nano) model is the smallest and
# fastest; swap in yolov8s.pt or yolov8m.pt for better accuracy at the cost of
# speed. Override with the MODEL_WEIGHTS environment variable.
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", "yolov8n.pt")

# Uploads bigger than this are rejected before we read them into memory.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 25 * 1024 * 1024))

# Very large photos are downscaled before inference. YOLO resizes internally
# anyway, and this keeps memory and upload-echo size sane. Detections are
# reported in the coordinates of the image the browser actually displays.
MAX_IMAGE_DIM = int(os.environ.get("MAX_IMAGE_DIM", 1920))

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Video mode. These are larger because even a short clip is a big file, but the
# real limits on cost are in video.py: frames are sampled down and capped.
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_BYTES", 200 * 1024 * 1024))

# The language assistant. The key itself comes from OPENAI_API_KEY in the
# environment and is deliberately never stored in this repository.
LLM_MODEL = os.environ.get("GEODETECT_LLM_MODEL", "gpt-5.4-mini")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8000))
