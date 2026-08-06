"""Minimal Grad-CAM++ placeholder for future explainability integration."""

from __future__ import annotations

import numpy as np


def generate_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim == 2:
        heatmap = np.stack([heatmap] * 3, axis=-1)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    overlay = (1 - alpha) * image.astype(np.float32) + alpha * (heatmap * 255)
    return np.clip(overlay, 0, 255).astype(np.uint8)
