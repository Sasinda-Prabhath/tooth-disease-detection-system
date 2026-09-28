from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.inference.model_loader import load_inference_model


class EnamelErosionEstimator:
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

    def predict(self, crop: np.ndarray) -> dict:
        if self.model is not None:
            return self._predict_model(crop)
        return self._predict_heuristic(crop)

    def _predict_model(self, crop: np.ndarray) -> dict:
        import tensorflow as tf

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        resized = cv2.resize(gray, (512, 512)).astype(np.float32) / 255.0
        inp = tf.convert_to_tensor(resized[None, ..., None], dtype=tf.float32)

        if self.model_kind == "savedmodel":
            out = self.model.signatures["serving_default"](inp)
            pred = list(out.values())[0]
        else:
            pred = self.model(inp, training=False)

        mask = pred.numpy()[0, ..., 0] if hasattr(pred, "numpy") else np.asarray(pred)[0, ..., 0]
        mask = (mask > 0.5).astype(np.float32)
        affected = float(mask.mean() * 100.0)
        label = self._grade_from_surface(affected)
        confidence = min(0.92, 0.55 + affected / 120.0)
        return {
            "label": label,
            "confidence": float(confidence),
            "affected_surface_pct": affected,
        }

    def _predict_heuristic(self, crop: np.ndarray) -> dict:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        bright = float((gray > 180).mean() * 100.0)
        texture = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        affected = min(100.0, bright * 0.4 + min(30.0, texture / 18.0))
        label = self._grade_from_surface(affected)
        confidence = min(0.85, 0.52 + affected / 130.0)
        return {
            "label": label,
            "confidence": float(confidence),
            "affected_surface_pct": float(affected),
        }

    @staticmethod
    def _grade_from_surface(affected_surface_pct: float) -> str:
        if affected_surface_pct >= 45:
            return "severe"
        if affected_surface_pct >= 25:
            return "moderate"
        if affected_surface_pct >= 10:
            return "mild"
        return "none"
