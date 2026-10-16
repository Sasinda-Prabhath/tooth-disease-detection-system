from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    yolo_model: str
    efficientnet_model: str
    device: str
    error: str | None = None


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    id: int
    bbox: BoundingBox
    yolo_confidence: float
    severity: str
    severity_confidence: float


class PredictionResponse(BaseModel):
    gingivitis_detected: bool
    status: str
    detection_count: int
    detections: list[Detection]
    summary: str
    most_severe_level: str | None = None
    image_width: int
    image_height: int
    raw_detection_count: int
    filtered_detection_count: int
    displayed_detection_count: int
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
