from __future__ import annotations

import os

import cv2
import numpy as np
from dental_common.image_io import load_image_bytes

from app.inference.exceptions import ImageQualityError


def assess_image_quality(rgb: np.ndarray) -> dict[str, float]:
    """Return objective pre-inference checks; thresholds are intentionally conservative."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {"width": float(gray.shape[1]), "height": float(gray.shape[0]), "contrast": float(gray.std()), "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())}


def require_acceptable_quality(rgb: np.ndarray) -> None:
    quality = assess_image_quality(rgb)
    if quality["width"] < 320 or quality["height"] < 160:
        raise ImageQualityError("Image resolution is too low. Upload a panoramic X-ray at least 320 x 160 pixels.")
    if quality["contrast"] < 4.0:
        raise ImageQualityError("Image contrast is too low for reliable X-ray analysis. Upload a clearer OPG image.")


def mask_metadata_bands(rgb: np.ndarray) -> np.ndarray:
    """Mask common edge/header metadata areas before inference.

    This protects obvious burned-in labels. It is not a substitute for an OCR
    de-identification service, which should be added for clinical deployment.
    """
    masked = rgb.copy()
    band = max(1, int(masked.shape[0] * 0.04))
    masked[:band] = 0
    masked[-band:] = 0
    return masked


def preprocess_panoramic(image_bytes: bytes) -> tuple[np.ndarray, float | None]:
    """CLAHE + denoise pipeline for panoramic X-rays. Returns RGB uint8 array."""
    rgb, pixel_spacing = load_image_bytes(image_bytes)
    require_acceptable_quality(rgb)
    rgb = mask_metadata_bands(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    rgb_out = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return rgb_out, pixel_spacing


def get_default_pixel_spacing() -> float:
    return float(os.getenv("PIXEL_SPACING_DEFAULT", "0.1"))
