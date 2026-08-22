from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import REQUIRE_TRAINED_MODELS
from app.inference.exceptions import ModelNotLoadedError
from app.inference.model_loader import load_inference_model

# FDI permanent dentition (11–18, 21–28, 31–38, 41–48)
FDI_PERMANENT = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
]

TOOTH_COLORS: dict[int, tuple[int, int, int]] = {
    fdi: (
        int(80 + (i * 37) % 175),
        int(100 + (i * 53) % 155),
        int(120 + (i * 71) % 135),
    )
    for i, fdi in enumerate(FDI_PERMANENT)
}


@dataclass
class ToothSegment:
    fdi_number: int
    bbox: tuple[float, float, float, float]
    confidence: float
    contour: list[list[float]]
    color_rgb: tuple[int, int, int]
    label: str


class ToothInstanceSegmenter:
    """Full-arch tooth instance segmentation (AKUDENTAL-style colored masks)."""

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

    def predict(self, image: np.ndarray, score_threshold: float = 0.25) -> list[ToothSegment]:
        if self.model is None:
            if REQUIRE_TRAINED_MODELS:
                raise ModelNotLoadedError("tooth_instance_segmenter")
            return []
        return self._predict_model(image, score_threshold)

    def _predict_model(self, image: np.ndarray, score_threshold: float) -> list[ToothSegment]:
        import tensorflow as tf

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        resized = cv2.resize(gray, (512, 512)).astype(np.float32) / 255.0
        inp = tf.convert_to_tensor(resized[None, ..., None], dtype=tf.float32)
        out = self._run_model(inp)
        logits = out.numpy()[0] if hasattr(out, "numpy") else np.asarray(out)[0]

        # Multi-class mask: (H, W, num_classes) softmax or (H, W) class indices
        if logits.ndim == 3:
            class_map = np.argmax(logits, axis=-1).astype(np.uint8)
        else:
            class_map = logits.astype(np.uint8)

        class_map = cv2.resize(class_map, (w, h), interpolation=cv2.INTER_NEAREST)
        segments: list[ToothSegment] = []

        for class_idx, fdi in enumerate(FDI_PERMANENT, start=1):
            mask = (class_map == class_idx).astype(np.uint8)
            if mask.sum() < 50:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < 100:
                continue

            x, y, bw, bh = cv2.boundingRect(largest)
            confidence = min(0.99, 0.5 + area / (w * h) * 10)
            if confidence < score_threshold:
                continue

            contour_pts = largest.reshape(-1, 2).astype(float).tolist()
            color = TOOTH_COLORS.get(fdi, (200, 200, 200))
            segments.append(
                ToothSegment(
                    fdi_number=fdi,
                    bbox=(float(x), float(y), float(x + bw), float(y + bh)),
                    confidence=float(confidence),
                    contour=contour_pts,
                    color_rgb=color,
                    label=_fdi_label(fdi),
                )
            )

        return segments

    def _run_model(self, inp) -> object:
        if self.model_kind == "savedmodel":
            if hasattr(self.model, "signatures") and "serving_default" in self.model.signatures:
                out = self.model.signatures["serving_default"](inp)
                return list(out.values())[0]
            return self.model(inp)
        return self.model(inp, training=False)


def _fdi_label(fdi: int) -> str:
    names = {
        11: "Central Incisor", 12: "Lateral Incisor", 13: "Canine", 14: "1st Premolar",
        15: "2nd Premolar", 16: "1st Molar", 17: "2nd Molar", 18: "3rd Molar",
        21: "Central Incisor", 22: "Lateral Incisor", 23: "Canine", 24: "1st Premolar",
        25: "2nd Premolar", 26: "1st Molar", 27: "2nd Molar", 28: "3rd Molar",
        31: "Central Incisor", 32: "Lateral Incisor", 33: "Canine", 34: "1st Premolar",
        35: "2nd Premolar", 36: "1st Molar", 37: "2nd Molar", 38: "3rd Molar",
        41: "Central Incisor", 42: "Lateral Incisor", 43: "Canine", 44: "1st Premolar",
        45: "2nd Premolar", 46: "1st Molar", 47: "2nd Molar", 48: "3rd Molar",
    }
    return f"{fdi} - {names.get(fdi, 'Tooth')}"
