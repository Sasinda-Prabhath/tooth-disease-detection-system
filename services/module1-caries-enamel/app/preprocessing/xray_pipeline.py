from __future__ import annotations

import os

import cv2
from dental_common.image_io import load_image_bytes


def preprocess_panoramic(image_bytes: bytes):
    """Preserve clinical-photo colors and enhance grayscale panoramic X-rays."""
    rgb, pixel_spacing = load_image_bytes(image_bytes)
    # The Enamel YOLO model is trained on color intraoral photographs. Applying
    # X-ray grayscale preprocessing to those images removes useful features.
    if _is_color_image(rgb):
        return rgb, pixel_spacing

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=8)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    rgb_out = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return rgb_out, pixel_spacing


def _is_color_image(rgb) -> bool:
    """Return true when the upload contains meaningful color information."""
    channels = rgb.astype("int16")
    channel_spread = channels.max(axis=2) - channels.min(axis=2)
    return float((channel_spread > 12).mean()) > 0.02


def get_default_pixel_spacing() -> float:
    return float(os.getenv("PIXEL_SPACING_DEFAULT", "0.1"))
