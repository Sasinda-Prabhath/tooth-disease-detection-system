from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.config import REQUIRE_TRAINED_MODELS
from app.inference.angulation import AngulationClassifier
from app.inference.bone_landmark import BoneLandmarkSegmenter
from app.inference.detector import MolarDetector
from app.inference.measurement import compute_periodontal_stage, measure_bone_loss, offset_landmarks_to_image
from app.inference.tooth_segmenter import ToothInstanceSegmenter
from app.schemas import AngulationResult, BoneLossResult, Module2Response, ToothResult, ToothSegmentationResult
from dental_common.schemas_base import BoundingBox

logger = logging.getLogger("module2.prediction")


@dataclass
class InferenceModels:
    detector: MolarDetector
    angulation: AngulationClassifier
    bone_segmenter: BoneLandmarkSegmenter
    tooth_segmenter: ToothInstanceSegmenter

    @property
    def status(self) -> dict[str, bool]:
        return {"molar_detector_yolov8": self.detector.is_loaded, "angulation_classifier": self.angulation.is_loaded, "bone_landmark_unet": self.bone_segmenter.is_loaded, "tooth_instance_segmenter": self.tooth_segmenter.is_loaded}

    def missing_required(self) -> list[str]:
        if not REQUIRE_TRAINED_MODELS:
            return []
        return [name for name, loaded in self.status.items() if not loaded]


class PredictionService:
    """Runs the trained-data Module 2 pipeline; YOLO detections are authoritative."""

    def __init__(self, models: InferenceModels):
        self.models = models

    @staticmethod
    def _bbox_tuple(bbox: BoundingBox) -> tuple[float, float, float, float]:
        return bbox.x1, bbox.y1, bbox.x2, bbox.y2

    def _analyze_bone(self, image: np.ndarray, bbox: BoundingBox, pixel_spacing_mm: float) -> BoneLossResult:
        roi = MolarDetector.crop(image, self._bbox_tuple(bbox))
        cej, crest = self.models.bone_segmenter.predict_landmarks(roi)
        return BoneLossResult(**measure_bone_loss(offset_landmarks_to_image(cej, bbox.x1, bbox.y1), offset_landmarks_to_image(crest, bbox.x1, bbox.y1), pixel_spacing_mm))

    def _segment_arch(self, image: np.ndarray) -> list[ToothSegmentationResult]:
        """Segmentation is annotation-only and never creates diagnostic teeth."""
        if not self.models.tooth_segmenter.is_loaded:
            return []
        segments: list[ToothSegmentationResult] = []
        for item in self.models.tooth_segmenter.predict(image):
            segments.append(ToothSegmentationResult(fdi_number=item.fdi_number, label=item.label, confidence=item.confidence, bbox=BoundingBox(x1=item.bbox[0], y1=item.bbox[1], x2=item.bbox[2], y2=item.bbox[3]), contour=item.contour, color_rgb=list(item.color_rgb)))
        return segments

    def predict(self, image: np.ndarray, pixel_spacing_mm: float, filename: str = "") -> Module2Response:
        missing = self.models.missing_required()
        if missing:
            from app.inference.exceptions import ModelNotLoadedError
            raise ModelNotLoadedError(missing[0])

        height, width = image.shape[:2]
        all_segments = self._segment_arch(image)
        detections = self.models.detector.predict(image)
        logger.info("YOLO third-molar detections for %s: %d", filename or "upload", len(detections))

        teeth: list[ToothResult] = []
        for detection in detections:
            bbox = BoundingBox(x1=detection.bbox[0], y1=detection.bbox[1], x2=detection.bbox[2], y2=detection.bbox[3])
            roi = MolarDetector.crop(image, detection.bbox)
            teeth.append(ToothResult(
                fdi_number=detection.fdi_number,
                bbox=bbox,
                is_third_molar=True,
                detection_confidence=detection.confidence,
                angulation=AngulationResult(**self.models.angulation.predict(roi)),
                bone_loss=self._analyze_bone(image, bbox, pixel_spacing_mm),
            ))

        bone_values = [tooth.bone_loss.bone_loss_mm for tooth in teeth if tooth.bone_loss]
        stage = compute_periodontal_stage(bone_values, teeth_detected=len(all_segments)) if bone_values else None
        return Module2Response(
            teeth=teeth,
            all_teeth_segments=all_segments,
            image_width=width,
            image_height=height,
            pixel_spacing_mm=pixel_spacing_mm,
            model_status=self.models.status,
            inference_mode="trained_yolov8",
            periodontal_stage=stage,
            mean_bone_loss_mm=round(float(np.mean(bone_values)), 2) if bone_values else None,
        )
