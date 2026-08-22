from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models_store"))

# When true, API rejects requests if any required model is missing (no heuristic fallback).
REQUIRE_TRAINED_MODELS = os.getenv("REQUIRE_TRAINED_MODELS", "1").lower() in ("1", "true", "yes")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
PIXEL_SPACING_DEFAULT = float(os.getenv("PIXEL_SPACING_DEFAULT", "0.1"))

REQUIRED_MODELS = (
    "molar_detector_yolov8",
    "angulation_classifier",
    "bone_landmark_unet",
    "tooth_instance_segmenter",
)
