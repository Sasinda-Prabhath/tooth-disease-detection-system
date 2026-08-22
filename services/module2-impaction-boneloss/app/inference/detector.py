"""YOLOv8 inference for FDI 18, 28, 38 and 48 third-molar detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import REQUIRE_TRAINED_MODELS
from app.inference.exceptions import ModelNotLoadedError

THIRD_MOLAR_FDI = {18, 28, 38, 48}
CLASS_TO_FDI = {0: 18, 1: 28, 2: 38, 3: 48}


@dataclass
class Detection:
    fdi_number: int
    bbox: tuple[float, float, float, float]
    confidence: float
    is_third_molar: bool = True


class MolarDetector:
    """Loads the YOLOv8 ``best.pt`` checkpoint trained from the project dataset.

    The model must use this class order: 0=18, 1=28, 2=38, 3=48.  No inferred
    boxes are manufactured when a trained model is required; this prevents a
    clinical report from containing results for teeth that YOLO did not detect.
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.weights_path = self.model_dir / "best.pt"
        self.model = None
        self._load()

    def _load(self) -> None:
        if not self.weights_path.exists():
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.weights_path))
        except ImportError as exc:
            raise RuntimeError("YOLOv8 is not installed. Install requirements.txt before starting the API.") from exc

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image: np.ndarray, score_threshold: float = 0.40) -> list[Detection]:
        if self.model is None:
            if REQUIRE_TRAINED_MODELS:
                raise ModelNotLoadedError("molar_detector")
            return []

        result = self.model.predict(source=image, imgsz=640, conf=score_threshold, verbose=False)[0]
        if result.boxes is None:
            return []

        detections: list[Detection] = []
        for box, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            fdi = CLASS_TO_FDI.get(int(class_id))
            if fdi is None:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            detections.append(Detection(fdi, (x1, y1, x2, y2), float(confidence)))

        # Keep one highest-confidence result per FDI class after YOLO NMS.
        best: dict[int, Detection] = {}
        for detection in detections:
            if detection.fdi_number not in best or detection.confidence > best[detection.fdi_number].confidence:
                best[detection.fdi_number] = detection
        return sorted(best.values(), key=lambda item: item.fdi_number)

    @staticmethod
    def crop(image: np.ndarray, bbox: tuple[float, float, float, float], padding_ratio: float = 0.0) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x, pad_y = (x2 - x1) * padding_ratio, (y2 - y1) * padding_ratio
        x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        x2, y2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
        crop = image[y1:y2, x1:x2]
        return crop if crop.size else np.zeros((224, 224, 3), dtype=np.uint8)
