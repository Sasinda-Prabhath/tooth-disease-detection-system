from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.inference.model_loader import load_inference_model

# FDI anchors used for a practical baseline detector fallback.
FDI_LIST = [16, 11, 26, 36, 31, 46]


@dataclass
class Detection:
    fdi_number: int
    bbox: tuple[float, float, float, float]
    confidence: float


class CariesToothDetector:
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

    def predict(self, image: np.ndarray, score_threshold: float = 0.25) -> list[Detection]:
        if self.model is not None:
            model_dets = self._predict_model(image, score_threshold)
            if model_dets:
                return model_dets
        return self._predict_heuristic(image)

    def _predict_model(self, image: np.ndarray, score_threshold: float) -> list[Detection]:
        import tensorflow as tf

        h, w = image.shape[:2]
        resized = cv2.resize(image, (640, 640)).astype(np.float32) / 255.0
        inp = tf.convert_to_tensor(resized[None, ...], dtype=tf.float32)

        if self.model_kind == "savedmodel":
            outputs = self.model.signatures["serving_default"](inp)
            boxes = outputs["boxes"].numpy()[0]
            classes = outputs["classes"].numpy()[0].astype(int)
            scores = outputs["scores"].numpy()[0]
        else:
            boxes, cls_probs = self.model(inp, training=False)
            boxes = boxes.numpy()[0].reshape(1, 4)
            cls_probs = cls_probs.numpy()[0]
            classes = np.array([int(np.argmax(cls_probs))])
            scores = np.array([float(np.max(cls_probs))])

        scale_x = w / 640.0
        scale_y = h / 640.0

        detections: list[Detection] = []
        for box, cls, score in zip(boxes, classes, scores):
            if float(score) < score_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            fdi = FDI_LIST[min(max(int(cls), 0), len(FDI_LIST) - 1)]
            detections.append(
                Detection(
                    fdi_number=fdi,
                    bbox=(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                    confidence=float(score),
                )
            )
        return detections

    def _predict_heuristic(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]

        regions = [
            (16, int(w * 0.08), int(h * 0.18), int(w * 0.24), int(h * 0.44)),
            (11, int(w * 0.34), int(h * 0.20), int(w * 0.46), int(h * 0.42)),
            (26, int(w * 0.74), int(h * 0.18), int(w * 0.90), int(h * 0.44)),
            (36, int(w * 0.08), int(h * 0.54), int(w * 0.24), int(h * 0.82)),
            (31, int(w * 0.34), int(h * 0.56), int(w * 0.46), int(h * 0.80)),
            (46, int(w * 0.74), int(h * 0.54), int(w * 0.90), int(h * 0.82)),
        ]

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detections: list[Detection] = []
        for fdi, x1, y1, x2, y2 in regions:
            roi = gray[y1:y2, x1:x2]
            contrast = float(roi.std()) if roi.size else 0.0
            score = min(0.9, 0.45 + contrast / 120.0)
            detections.append(
                Detection(
                    fdi_number=fdi,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=score,
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
