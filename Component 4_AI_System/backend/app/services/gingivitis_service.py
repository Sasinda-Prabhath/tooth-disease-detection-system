from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "gingivitis" / "gingivitis_yolo26s_final_best.pt"
EFFICIENTNET_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gingivitis"
    / "efficientnet_b0_stage2_severity_best.pth"
)

YOLO_CONFIDENCE_THRESHOLD = 0.45
YOLO_NMS_IOU_THRESHOLD = 0.45
DUPLICATE_IOU_THRESHOLD = 0.60
MAX_DETECTIONS = 12
MIN_CONFIDENCE_THRESHOLD = 0.10
MAX_CONFIDENCE_THRESHOLD = 0.90
MIN_YOLO_NMS_IOU_THRESHOLD = 0.20
MAX_YOLO_NMS_IOU_THRESHOLD = 0.80
MIN_MAX_DETECTIONS = 1
MAX_MAX_DETECTIONS = 30
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = {
    0: "Mild",
    1: "Moderate",
    2: "Severe",
}
SEVERITY_RANK = {
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
}

EFFICIENTNET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class ModelLoadError(RuntimeError):
    pass


class PredictionError(RuntimeError):
    pass


def select_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_efficientnet_b0(num_classes: int = 3) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("EfficientNet checkpoint does not contain a valid state_dict.")
    return state_dict


def validate_model_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} path is not a file: {path}")


def load_image_from_bytes(image_bytes: bytes, filename: str) -> Image.Image:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported image format. Use one of: {supported}")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc
    except OSError as exc:
        raise ValueError("Uploaded image could not be opened.") from exc


def clamp_box(
    box_xyxy: list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box_xyxy
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second

    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection == 0:
        return 0.0

    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def remove_near_duplicate_boxes(
    candidates: list[dict[str, Any]],
    max_detections: int,
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: item["yolo_confidence"],
        reverse=True,
    ):
        if any(
            box_iou(candidate["box"], existing["box"]) > DUPLICATE_IOU_THRESHOLD
            for existing in retained
        ):
            continue
        retained.append(candidate)
        if len(retained) >= max_detections:
            break
    return retained


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return min(max(value, minimum), maximum)


class GingivitisHybridModelService:
    def __init__(self) -> None:
        self.device = select_device()
        self.yolo_model: YOLO | None = None
        self.efficientnet_model: nn.Module | None = None
        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.yolo_model is not None and self.efficientnet_model is not None

    def load_models(self) -> None:
        self.load_error = None

        try:
            print(f"[model-load] project root: {PROJECT_ROOT}", flush=True)
            print(f"[model-load] YOLO model path: {YOLO_MODEL_PATH}", flush=True)
            print(
                f"[model-load] EfficientNet model path: {EFFICIENTNET_MODEL_PATH}",
                flush=True,
            )
            validate_model_path(YOLO_MODEL_PATH, "YOLO")
            validate_model_path(EFFICIENTNET_MODEL_PATH, "EfficientNet")

            self.yolo_model = YOLO(str(YOLO_MODEL_PATH))

            efficientnet_model = build_efficientnet_b0(num_classes=len(CLASS_NAMES))
            checkpoint = torch.load(
                EFFICIENTNET_MODEL_PATH,
                map_location=self.device,
                weights_only=False,
            )
            state_dict = extract_state_dict(checkpoint)
            efficientnet_model.load_state_dict(state_dict, strict=True)
            efficientnet_model.to(self.device)
            efficientnet_model.eval()
            self.efficientnet_model = efficientnet_model
        except Exception as exc:
            self.yolo_model = None
            self.efficientnet_model = None
            self.load_error = str(exc)
            raise ModelLoadError(str(exc)) from exc

    def health(self) -> dict[str, str | None]:
        return {
            "status": "ok" if self.is_loaded else "error",
            "yolo_model": "loaded" if self.yolo_model is not None else "not_loaded",
            "efficientnet_model": (
                "loaded" if self.efficientnet_model is not None else "not_loaded"
            ),
            "device": str(self.device),
            "error": self.load_error,
        }

    def classify_roi(self, roi: Image.Image) -> tuple[str, float]:
        if self.efficientnet_model is None:
            raise PredictionError("EfficientNet model is not loaded.")

        tensor = EFFICIENTNET_TRANSFORM(roi).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.efficientnet_model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_index = torch.max(probabilities, dim=0)

        class_id = int(class_index.item())
        return CLASS_NAMES[class_id], float(confidence.item())

    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold: float = YOLO_NMS_IOU_THRESHOLD,
        max_detections: int = MAX_DETECTIONS,
    ) -> dict[str, Any]:
        if not self.is_loaded or self.yolo_model is None:
            raise PredictionError(
                self.load_error or "Models are not loaded. Check /health."
            )

        confidence_threshold = clamp_float(
            confidence_threshold,
            MIN_CONFIDENCE_THRESHOLD,
            MAX_CONFIDENCE_THRESHOLD,
        )
        iou_threshold = clamp_float(
            iou_threshold,
            MIN_YOLO_NMS_IOU_THRESHOLD,
            MAX_YOLO_NMS_IOU_THRESHOLD,
        )
        max_detections = clamp_int(
            max_detections,
            MIN_MAX_DETECTIONS,
            MAX_MAX_DETECTIONS,
        )

        image = load_image_from_bytes(image_bytes, filename)
        width, height = image.size

        try:
            results = self.yolo_model.predict(
                source=np.array(image),
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False,
                device=0 if self.device.type == "cuda" else "cpu",
            )
        except Exception as exc:
            raise PredictionError(f"YOLO inference failed: {exc}") from exc

        raw_candidates: list[dict[str, Any]] = []
        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                for box in boxes:
                    yolo_confidence = float(box.conf.detach().cpu().item())
                    raw_box = box.xyxy.detach().cpu().numpy()[0].tolist()
                    safe_box = clamp_box(raw_box, width=width, height=height)
                    if safe_box is None:
                        continue

                    raw_candidates.append(
                        {
                            "box": safe_box,
                            "yolo_confidence": yolo_confidence,
                        }
                    )

        raw_candidates = sorted(
            raw_candidates,
            key=lambda item: item["yolo_confidence"],
            reverse=True,
        )
        filtered_candidates = remove_near_duplicate_boxes(
            raw_candidates,
            max_detections=max_detections,
        )
        filtered_candidates = sorted(
            filtered_candidates,
            key=lambda item: item["yolo_confidence"],
            reverse=True,
        )[:max_detections]
        print(f"[/predict] raw detections count: {len(raw_candidates)}", flush=True)
        print(
            f"[/predict] filtered detections count: {len(filtered_candidates)}",
            flush=True,
        )

        detections: list[dict[str, Any]] = []
        for candidate in filtered_candidates:
            roi = image.crop(candidate["box"])
            severity, severity_confidence = self.classify_roi(roi)
            x1, y1, x2, y2 = candidate["box"]
            detections.append(
                {
                    "id": len(detections) + 1,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "yolo_confidence": round(candidate["yolo_confidence"], 6),
                    "severity": severity,
                    "severity_confidence": round(severity_confidence, 6),
                }
            )

        detections = sorted(
            detections,
            key=lambda item: item["yolo_confidence"],
            reverse=True,
        )[:max_detections]
        for index, detection in enumerate(detections, start=1):
            detection["id"] = index

        assert len(detections) <= max_detections

        if not detections:
            return {
                "gingivitis_detected": False,
                "status": "Healthy appearance",
                "detection_count": 0,
                "detections": [],
                "summary": "No confident gingivitis regions detected",
                "most_severe_level": None,
                "image_width": width,
                "image_height": height,
                "raw_detection_count": len(raw_candidates),
                "filtered_detection_count": len(filtered_candidates),
                "displayed_detection_count": len(detections),
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "max_detections": max_detections,
            }

        most_severe_level = max(
            (detection["severity"] for detection in detections),
            key=lambda severity: SEVERITY_RANK[severity],
        )

        return {
            "gingivitis_detected": True,
            "status": "Gingivitis detected",
            "detection_count": len(detections),
            "detections": detections,
            "summary": f"{len(detections)} gingivitis region(s) detected",
            "most_severe_level": most_severe_level,
            "image_width": width,
            "image_height": height,
            "raw_detection_count": len(raw_candidates),
            "filtered_detection_count": len(filtered_candidates),
            "displayed_detection_count": len(detections),
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "max_detections": max_detections,
        }
