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
        self.model_path = None
        self.yolo_model = None
        self.yolo_models: list[tuple[str, object, Path]] = []
        self.fasterrcnn_model = None
        self._load()

    def _load(self) -> None:
        yolo_candidates = (
            self.model_dir / "enamel_erosion_yolo26s.pt",
            self.model_dir / "yolo26_enamel_erosion_best.pt",
            self.model_dir / "best.pt",
        )
        faster_rcnn_path = self.model_dir / "faster_rcnn_dental_erosion.pth"

        existing_yolo_paths = [path for path in yolo_candidates if path.exists()]
        if existing_yolo_paths:
            from ultralytics import YOLO

            for yolo_path in existing_yolo_paths:
                model = YOLO(str(yolo_path))
                model_name = {
                    "enamel_erosion_yolo26s.pt": "Enamel YOLO26s",
                    "yolo26_enamel_erosion_best.pt": "Enamel YOLO26",
                    "best.pt": "Enamel YOLO",
                }.get(yolo_path.name, f"Enamel YOLO ({yolo_path.stem})")
                self.yolo_models.append((model_name, model, yolo_path))

            self.yolo_model = self.yolo_models[0][1]
            self.model = self.yolo_model
            self.model_kind = "ultralytics"
            self.model_path = self.yolo_models[0][2]

        if faster_rcnn_path.exists():
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn

            checkpoint = torch.load(faster_rcnn_path, map_location="cpu", weights_only=True)
            class_count = int(checkpoint["roi_heads.box_predictor.cls_score.weight"].shape[0])
            model = fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                num_classes=class_count,
            )
            model.load_state_dict(checkpoint)
            model.eval()
            self.fasterrcnn_model = model
            if self.model is None:
                self.model = model
                self.model_kind = "fasterrcnn"
                self.model_path = faster_rcnn_path

        if self.model is None:
            self.model, self.model_kind = load_inference_model(self.model_dir)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, crop: np.ndarray) -> dict:
        if self.model is not None:
            return self._predict_model(crop)
        return self._predict_heuristic(crop)

    def predict_models(self, image: np.ndarray) -> list[dict]:
        outputs = []
        for model_name, model, _ in self.yolo_models:
            result, boxes = self._predict_ultralytics_with_boxes(image, model)
            outputs.append({"model": model_name, "result": result, "boxes": boxes})
        if self.fasterrcnn_model is not None:
            result, boxes = self._predict_fasterrcnn_with_boxes(image, self.fasterrcnn_model)
            outputs.append({"model": "Enamel Faster R-CNN", "result": result, "boxes": boxes})
        if not outputs and self.model is not None:
            outputs.append({"model": "Enamel model", "result": self.predict(image), "boxes": []})
        return outputs

    def _predict_model(self, crop: np.ndarray) -> dict:
        if self.model_kind == "ultralytics":
            return self._predict_ultralytics(crop)
        if self.model_kind == "fasterrcnn":
            return self._predict_fasterrcnn(crop)

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

    def _predict_ultralytics(self, crop: np.ndarray) -> dict:
        result, _ = self._predict_ultralytics_with_boxes(crop, self.model)
        return result

    def _predict_ultralytics_with_boxes(self, crop: np.ndarray, model) -> tuple[dict, list[dict]]:
        results = model.predict(source=crop, conf=0.25, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return self._erosion_result(0.0, 0.0), []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        height, width = crop.shape[:2]
        image_area = max(float(height * width), 1.0)
        detected_area = 0.0
        annotations = []
        for (x1, y1, x2, y2), score, class_id in zip(boxes, scores, classes):
            clipped_x1 = max(0.0, min(float(x1), width))
            clipped_y1 = max(0.0, min(float(y1), height))
            clipped_x2 = max(clipped_x1, min(float(x2), width))
            clipped_y2 = max(clipped_y1, min(float(y2), height))
            detected_area += (clipped_x2 - clipped_x1) * (clipped_y2 - clipped_y1)
            annotations.append({
                "x1": clipped_x1, "y1": clipped_y1,
                "x2": clipped_x2, "y2": clipped_y2,
                "confidence": float(score), "label": "enamel erosion",
                "class_id": int(class_id),
            })

        affected = min(100.0, detected_area / image_area * 100.0)
        return self._erosion_result(affected, float(np.max(scores))), annotations

    def _predict_fasterrcnn(self, crop: np.ndarray) -> dict:
        result, _ = self._predict_fasterrcnn_with_boxes(crop, self.model)
        return result

    def _predict_fasterrcnn_with_boxes(self, crop: np.ndarray, model) -> tuple[dict, list[dict]]:
        import torch

        image = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            prediction = model([image])[0]

        scores = prediction["scores"].cpu().numpy()
        boxes = prediction["boxes"].cpu().numpy()
        keep = scores >= 0.25
        if not np.any(keep):
            return self._erosion_result(0.0, 0.0), []

        height, width = crop.shape[:2]
        image_area = max(float(height * width), 1.0)
        detected_area = 0.0
        annotations = []
        for (x1, y1, x2, y2), score in zip(boxes[keep], scores[keep]):
            clipped_x1 = max(0.0, min(float(x1), width))
            clipped_y1 = max(0.0, min(float(y1), height))
            clipped_x2 = max(clipped_x1, min(float(x2), width))
            clipped_y2 = max(clipped_y1, min(float(y2), height))
            detected_area += (clipped_x2 - clipped_x1) * (clipped_y2 - clipped_y1)
            annotations.append({
                "x1": clipped_x1, "y1": clipped_y1,
                "x2": clipped_x2, "y2": clipped_y2,
                "confidence": float(score), "label": "enamel erosion",
            })

        affected = min(100.0, detected_area / image_area * 100.0)
        return self._erosion_result(affected, float(np.max(scores[keep]))), annotations

    def _erosion_result(self, affected: float, confidence: float) -> dict:
        return {
            "label": self._grade_from_surface(affected),
            "confidence": float(confidence),
            "affected_surface_pct": float(affected),
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
