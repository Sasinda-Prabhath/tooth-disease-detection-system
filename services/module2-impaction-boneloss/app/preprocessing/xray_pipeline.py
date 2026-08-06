from __future__ import annotations

import os

import cv2
import numpy as np
from dental_common.image_io import load_image_bytes


def preprocess_panoramic(image_bytes: bytes) -> tuple[np.ndarray, float | None]:
    """CLAHE + denoise pipeline for panoramic X-rays. Returns RGB uint8 array."""
    rgb, pixel_spacing = load_image_bytes(image_bytes)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    rgb_out = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return rgb_out, pixel_spacing


def get_default_pixel_spacing() -> float:
    return float(os.getenv("PIXEL_SPACING_DEFAULT", "0.1"))
