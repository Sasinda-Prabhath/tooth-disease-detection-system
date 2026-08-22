from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import MODELS_DIR
from app.controllers.predict_controller import create_predict_router
from app.inference.angulation import AngulationClassifier
from app.inference.bone_landmark import BoneLandmarkSegmenter
from app.inference.detector import MolarDetector
from app.inference.tooth_segmenter import ToothInstanceSegmenter
from app.logging_config import configure_logging
from app.services.prediction_service import InferenceModels

configure_logging()
logger = logging.getLogger("module2.main")

app = FastAPI(
    title="Module 2 — Impaction & Bone Loss Service",
    description="Impacted third molar angulation and alveolar bone loss detection from panoramic X-rays.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


_models = InferenceModels(
    detector=MolarDetector(MODELS_DIR / "molar_detector"),
    angulation=AngulationClassifier(MODELS_DIR / "angulation_classifier"),
    bone_segmenter=BoneLandmarkSegmenter(MODELS_DIR / "bone_landmark_unet"),
    tooth_segmenter=ToothInstanceSegmenter(MODELS_DIR / "tooth_instance_segmenter"),
)

app.include_router(create_predict_router(_models))


@app.on_event("startup")
def log_startup() -> None:
    status = _models.status
    logger.info("Module 2 API ready | models=%s", status)