from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import REQUIRE_TRAINED_MODELS
from app.inference.exceptions import ModelNotLoadedError
from app.inference.model_loader import load_inference_model


class BoneLandmarkSegmenter:
    """U-Net segmenter for CEJ and alveolar crest landmark lines."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model = None
        self.model_kind = None
        self._load()

    def _load(self) -> None:
        self.model, self.model_kind = load_inference_model(self.model_dir)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_landmarks(self, crop: np.ndarray) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        if self.model is not None:
            return self._predict_model(crop)
        if REQUIRE_TRAINED_MODELS:
            raise ModelNotLoadedError("bone_landmark_unet")
        return self._predict_heuristic(crop)

    def _predict_model(self, crop: np.ndarray) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        import tensorflow as tf

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        resized = cv2.resize(gray, (512, 512)).astype(np.float32) / 255.0
        inp = tf.convert_to_tensor(resized[None, ..., None], dtype=tf.float32)
        out = self._run_model(inp)
        masks = out.numpy()[0] if hasattr(out, "numpy") else np.asarray(out)[0]

        h, w = crop.shape[:2]
        cej_mask = cv2.resize(masks[..., 0], (w, h))
        crest_mask = cv2.resize(masks[..., 1], (w, h))

        cej_points = self._mask_to_line_points(cej_mask)
        crest_points = self._mask_to_line_points(crest_mask)
        return cej_points, crest_points

    def _run_model(self, inp) -> object:
        if self.model_kind == "savedmodel":
            if hasattr(self.model, "signatures") and "serving_default" in self.model.signatures:
                out = self.model.signatures["serving_default"](inp)
                return list(out.values())[0]
            return self.model(inp)
        return self.model(inp, training=False)

    def _predict_heuristic(
        self, crop: np.ndarray
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        h, w = crop.shape[:2]
        mid_x = w * 0.5
        cej_y = h * 0.55
        crest_y = h * 0.35
        cej_points = [(mid_x - w * 0.15, cej_y), (mid_x + w * 0.15, cej_y)]
        crest_points = [(mid_x - w * 0.15, crest_y), (mid_x + w * 0.15, crest_y)]
        return cej_points, crest_points

    @staticmethod
    def _mask_to_line_points(mask: np.ndarray, num_points: int = 2) -> list[tuple[float, float]]:
        """Extract a horizontal landmark line (mesial–distal) at the mask median row."""
        ys, xs = np.where(mask > 0.5)
        h, w = mask.shape
        if len(xs) == 0:
            y = h * 0.5
            return [(w * 0.1, y), (w * 0.9, y)]

        y_med = float(np.median(ys))
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        if num_points == 2:
            return [(x_min, y_med), (x_max, y_med)]

        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        idx = [0, len(xs) - 1]
        return [(float(xs[i]), float(ys[i])) for i in idx]
