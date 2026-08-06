from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.inference.angulation import AngulationClassifier
from app.inference.bone_landmark import BoneLandmarkSegmenter
from app.inference.detector import MolarDetector
from app.inference.measurement import measure_bone_loss
from app.preprocessing.xray_pipeline import get_default_pixel_spacing, preprocess_panoramic
from app.schemas import (
    AngulationResult,
    BoneLossResult,
    Module2HealthResponse,
    Module2Response,
    ToothResult,
)
from dental_common.schemas_base import BoundingBox

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models_store"))

app = FastAPI(
    title="Module 2 — Impaction & Bone Loss Service",
    description="Impacted third molar angulation and alveolar bone loss detection from panoramic X-rays.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = MolarDetector(MODELS_DIR / "molar_detector")
angulation_clf = AngulationClassifier(MODELS_DIR / "angulation_classifier")
bone_segmenter = BoneLandmarkSegmenter(MODELS_DIR / "bone_landmark_unet")


@app.get("/health", response_model=Module2HealthResponse)
def health() -> Module2HealthResponse:
    models = {
        "molar_detector": detector.is_loaded,
        "angulation_classifier": angulation_clf.is_loaded,
        "bone_landmark_unet": bone_segmenter.is_loaded,
    }
    return Module2HealthResponse(
        status="ok",
        service="module2-impaction-boneloss",
        models_loaded=models,
    )


@app.post("/predict", response_model=Module2Response)
async def predict(
    file: UploadFile = File(...),
    pixel_spacing_mm: float | None = Form(default=None),
) -> Module2Response:
    image_bytes = await file.read()
    processed, dicom_spacing = preprocess_panoramic(image_bytes)
    spacing = pixel_spacing_mm or dicom_spacing or get_default_pixel_spacing()

    detections = detector.predict(processed)
    teeth: list[ToothResult] = []

    for tooth in detections:
        entry = ToothResult(
            fdi_number=tooth.fdi_number,
            bbox=BoundingBox(
                x1=tooth.bbox[0],
                y1=tooth.bbox[1],
                x2=tooth.bbox[2],
                y2=tooth.bbox[3],
            ),
            is_third_molar=tooth.is_third_molar,
        )

        if tooth.is_third_molar:
            crop = MolarDetector.crop(processed, tooth.bbox)
            ang = angulation_clf.predict(crop)
            entry.angulation = AngulationResult(**ang)

            cej, crest = bone_segmenter.predict_landmarks(crop)
            bone = measure_bone_loss(cej, crest, spacing)
            entry.bone_loss = BoneLossResult(**bone)

        teeth.append(entry)

    h, w = processed.shape[:2]
    return Module2Response(
        teeth=teeth,
        image_width=w,
        image_height=h,
        pixel_spacing_mm=spacing,
        model_status={
            "molar_detector": detector.is_loaded,
            "angulation_classifier": angulation_clf.is_loaded,
            "bone_landmark_unet": bone_segmenter.is_loaded,
        },
    )
