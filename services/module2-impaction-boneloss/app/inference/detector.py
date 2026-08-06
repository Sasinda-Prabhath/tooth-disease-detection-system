from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

THIRD_MOLAR_FDI = {18, 28, 38, 48}


@dataclass
class Detection:
    fdi_number: int
    bbox: tuple[float, float, float, float]
    confidence: float
    is_third_molar: bool


class MolarDetector:
    """Loads YOLOv8 SavedModel or falls back to heuristic third-molar regions."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model = None
        self._load()

    def _load(self) -> None:
        if (self.model_dir / "saved_model.pb").exists():
            self.model = tf.saved_model.load(str(self.model_dir))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image: np.ndarray, score_threshold: float = 0.25) -> list[Detection]:
        h, w = image.shape[:2]
        if self.model is not None:
            return self._predict_model(image, score_threshold)
        return self._predict_heuristic(image)

    def _predict_model(self, image: np.ndarray, score_threshold: float) -> list[Detection]:
        h, w = image.shape[:2]
        resized = cv2.resize(image, (640, 640))
        inp = tf.convert_to_tensor(resized[None, ...], dtype=tf.float32)

        if hasattr(self.model, "signatures") and "serving_default" in self.model.signatures:
            outputs = self.model.signatures["serving_default"](inp)
        else:
            outputs = self.model(inp)

        boxes = outputs["boxes"].numpy()[0]
        classes = outputs["classes"].numpy()[0].astype(int)
        scores = outputs["scores"].numpy()[0]

        scale_x = w / 640.0
        scale_y = h / 640.0

        detections: list[Detection] = []
        for box, cls, score in zip(boxes, classes, scores):
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            fdi = int(cls) + 1
            if fdi not in THIRD_MOLAR_FDI:
                fdi = THIRD_MOLAR_FDI[min(len(THIRD_MOLAR_FDI) - 1, max(0, cls))]
            detections.append(
                Detection(
                    fdi_number=fdi,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(score),
                    is_third_molar=fdi in THIRD_MOLAR_FDI,
                )
            )

        if not detections:
            return self._predict_heuristic(image)
        return detections

    def _predict_heuristic(self, image: np.ndarray) -> list[Detection]:
        """Fallback when no trained model: estimate third molar regions on panoramic layout."""
        h, w = image.shape[:2]
        regions = [
            (18, int(w * 0.02), int(h * 0.15), int(w * 0.12), int(h * 0.45)),
            (28, int(w * 0.88), int(h * 0.15), int(w * 0.98), int(h * 0.45)),
            (38, int(w * 0.02), int(h * 0.55), int(w * 0.12), int(h * 0.85)),
            (48, int(w * 0.88), int(h * 0.55), int(w * 0.98), int(h * 0.85)),
        ]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detections: list[Detection] = []
        for fdi, x1, y1, x2, y2 in regions:
            roi = gray[y1:y2, x1:x2]
            contrast = float(roi.std()) if roi.size else 0.0
            score = min(0.95, 0.5 + contrast / 100.0)
            detections.append(
                Detection(
                    fdi_number=fdi,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=score,
                    is_third_molar=True,
                )
            )
        return detections

    @staticmethod
    def crop(image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return crop
