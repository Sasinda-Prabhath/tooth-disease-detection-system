from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.inference.model_loader import load_inference_model
from app.schemas import ANGULATION_LABELS


class AngulationClassifier:
    """EfficientNetV2B0 angulation classifier for impacted third molars."""

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

        resized = cv2.resize(crop, (224, 224))
        inp = tf.convert_to_tensor(resized[None, ...], dtype=tf.float32)
        out = self._run_model(inp)
        probs = out.numpy()[0] if hasattr(out, "numpy") else np.asarray(out)[0]
        idx = int(np.argmax(probs))
        return {
            "label": ANGULATION_LABELS[idx],
            "confidence": float(probs[idx]),
            "is_impacted": True,
        }

    def _run_model(self, inp) -> object:
        import tensorflow as tf

        if self.model_kind == "savedmodel":
            if hasattr(self.model, "signatures") and "serving_default" in self.model.signatures:
                out = self.model.signatures["serving_default"](inp)
                return list(out.values())[0]
            return self.model(inp)
        return self.model(inp, training=False)

    def _predict_heuristic(self, crop: np.ndarray) -> dict:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        h, w = gray.shape[:2]
        aspect = w / max(h, 1)
        if aspect > 1.3:
            label, conf = "Horizontal", 0.72
        elif aspect < 0.7:
            label, conf = "Vertical", 0.68
        else:
            left = gray[:, : w // 2].mean()
            right = gray[:, w // 2 :].mean()
            label = "Mesioangular" if left > right else "Distoangular"
            conf = 0.65
        return {"label": label, "confidence": conf, "is_impacted": True}
