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
enamel_estimator = EnamelErosionEstimator(MODELS_DIR / "enamel_erosion_unet")


@app.get("/health")
def health() -> Module1HealthResponse:
    models = {
        "caries_detector": detector.is_loaded,
        "caries_classifier": caries_clf.is_loaded,
        "enamel_erosion_unet": enamel_estimator.is_loaded,
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
    teeth: list[ToothResult] = []

    for det in detections:
        crop = detector.crop(processed, det.bbox)
        caries = caries_clf.predict(crop)
        enamel = enamel_estimator.predict(crop)
        teeth.append(
            ToothResult(
                fdi_number=det.fdi_number,
                bbox=BoundingBox(
                    x1=det.bbox[0],
                    y1=det.bbox[1],
                    x2=det.bbox[2],
                    y2=det.bbox[3],
                ),
                caries=caries,
                enamel_erosion=enamel,
            )
        )

    return Module1Response(
        teeth=teeth,
        image_width=int(processed.shape[1]),
        image_height=int(processed.shape[0]),
        pixel_spacing_mm=float(spacing),
        model_status={
            "caries_detector": detector.is_loaded,
            "caries_classifier": caries_clf.is_loaded,
            "enamel_erosion_unet": enamel_estimator.is_loaded,
        },
    )
