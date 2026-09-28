from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.inference.model_loader import load_inference_model
from app.schemas import CARIES_LABELS


class CariesClassifier:
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

        resized = cv2.resize(crop, (224, 224)).astype(np.float32)
        inp = tf.convert_to_tensor(resized[None, ...], dtype=tf.float32)

        if self.model_kind == "savedmodel":
            out = self.model.signatures["serving_default"](inp)
            tensor = list(out.values())[0]
        else:
            tensor = self.model(inp, training=False)

        probs = tensor.numpy()[0] if hasattr(tensor, "numpy") else np.asarray(tensor)[0]
        idx = int(np.argmax(probs))
        return {
            "label": CARIES_LABELS[idx],
            "confidence": float(probs[idx]),
        }

    def _predict_heuristic(self, crop: np.ndarray) -> dict:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        low_intensity = float((gray < 55).mean())
        edge_density = float(cv2.Canny(gray, 70, 160).mean() / 255.0)
        risk = 0.65 * low_intensity + 0.35 * edge_density

        if risk > 0.22:
            label, conf = "severe", min(0.88, 0.55 + risk)
        elif risk > 0.14:
            label, conf = "moderate", min(0.82, 0.5 + risk)
        elif risk > 0.08:
            label, conf = "incipient", min(0.76, 0.45 + risk)
        else:
            label, conf = "none", max(0.55, 0.8 - risk)

        return {"label": label, "confidence": float(conf)}
