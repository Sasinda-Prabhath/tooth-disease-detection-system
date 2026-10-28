from __future__ import annotations
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.inference.caries import CariesClassifier
from app.inference.detector import CariesToothDetector
from app.inference.enamel import EnamelErosionEstimator
from app.preprocessing.xray_pipeline import get_default_pixel_spacing, preprocess_panoramic
from app.schemas import Module1HealthResponse, Module1Response, ToothResult
from dental_common.schemas_base import BoundingBox

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models_store"))

app = FastAPI(
    title="Module 1 - Caries and Enamel Service",
    description="Caries and enamel erosion screening from panoramic X-rays.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = CariesToothDetector(MODELS_DIR / "caries_detector")
caries_clf = CariesClassifier(MODELS_DIR / "caries_classifier")
enamel_estimator = EnamelErosionEstimator(MODELS_DIR / "enamel_erosion_detector")


@app.get("/health")
def health() -> Module1HealthResponse:
    models = {
        "caries_detector": detector.is_loaded,
        "caries_classifier": caries_clf.is_loaded,
        "enamel_yolo26s": any(path.name == "enamel_erosion_yolo26s.pt" for _, _, path in enamel_estimator.yolo_models),
        "enamel_yolo26": any(path.name == "yolo26_enamel_erosion_best.pt" for _, _, path in enamel_estimator.yolo_models),
        "enamel_faster_rcnn": enamel_estimator.fasterrcnn_model is not None,
    }
    return Module1HealthResponse(
        status="ok",
        service="module1-caries-enamel",
        models_loaded=models,
    )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    pixel_spacing_mm: float | None = Form(default=None),
) -> Module1Response:
    image_bytes = await file.read()
    processed, dicom_spacing = preprocess_panoramic(image_bytes)
    spacing = pixel_spacing_mm or dicom_spacing or get_default_pixel_spacing()

    detections = detector.predict(processed)
    enamel_models = enamel_estimator.predict_models(processed)
    primary_enamel_model = max(
        enamel_models,
        key=lambda model: model["result"].get("confidence", 0.0),
        default=None,
    )
    enamel_result = primary_enamel_model["result"] if primary_enamel_model else None
    teeth: list[ToothResult] = []
    annotations: list[dict] = []
    caries_confidences = []

    for det in detections:
        crop = detector.crop(processed, det.bbox)
        caries = caries_clf.predict(crop)
        caries_confidences.append(caries["confidence"])
        annotations.append({
            "x1": det.bbox[0], "y1": det.bbox[1],
            "x2": det.bbox[2], "y2": det.bbox[3],
            "confidence": det.confidence,
            "label": f"FDI {det.fdi_number} · caries",
            "source": "Dental caries detector",
        })
        teeth.append(
            ToothResult(
                fdi_number=det.fdi_number,
                bbox=BoundingBox(
                    x1=det.bbox[0],
                    y1=det.bbox[1],
                    x2=det.bbox[2],
                    y2=det.bbox[3],
                ),
                yolo_confidence=det.confidence,
                caries=caries,
                # The erosion detector is trained on the complete clinical
                # image, so its image-level result is returned below rather
                # than incorrectly rerunning it on a tooth crop.
                enamel_erosion=None,
            )
        )

    model_results = {
        "Dental caries detector": {
            "type": "tooth detection",
            "detections": len(detections),
            "average_confidence": float(sum(det.confidence for det in detections) / len(detections)) if detections else 0.0,
        },
        "Dental caries classifier": {
            "type": "severity classification",
            "predictions": len(caries_confidences),
            "average_confidence": float(sum(caries_confidences) / len(caries_confidences)) if caries_confidences else 0.0,
        },
    }
    for enamel_model in enamel_models:
        result = enamel_model["result"]
        model_results[enamel_model["model"]] = {
            "type": "image-level erosion detection",
            **result,
        }
        for box in enamel_model["boxes"]:
            annotations.append({**box, "source": enamel_model["model"]})

    return Module1Response(
        teeth=teeth,
        enamel_erosion=enamel_result,
        image_width=int(processed.shape[1]),
        image_height=int(processed.shape[0]),
        pixel_spacing_mm=float(spacing),
        model_status={
            "caries_detector": detector.is_loaded,
            "caries_classifier": caries_clf.is_loaded,
            "enamel_yolo26s": any(path.name == "enamel_erosion_yolo26s.pt" for _, _, path in enamel_estimator.yolo_models),
            "enamel_yolo26": any(path.name == "yolo26_enamel_erosion_best.pt" for _, _, path in enamel_estimator.yolo_models),
            "enamel_faster_rcnn": enamel_estimator.fasterrcnn_model is not None,
        },
        model_results=model_results,
        annotations=annotations,
    )
