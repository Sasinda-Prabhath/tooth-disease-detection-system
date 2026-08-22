from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.inference.exceptions import ImageQualityError, ModelNotLoadedError
from app.preprocessing.xray_pipeline import get_default_pixel_spacing, preprocess_panoramic
from app.schemas import Module2HealthResponse, Module2Response
from app.services.prediction_service import InferenceModels, PredictionService

logger = logging.getLogger("module2.controller")

router = APIRouter()


def create_predict_router(models: InferenceModels) -> APIRouter:
    service = PredictionService(models)

    @router.get("/health", response_model=Module2HealthResponse)
    def health() -> Module2HealthResponse:
        status = models.status
        all_loaded = all(status.values())
        return Module2HealthResponse(
            status="ok" if all_loaded else "degraded",
            service="module2-impaction-boneloss",
            models_loaded=status,
        )

    @router.post("/predict", response_model=Module2Response)
    async def predict(
        file: UploadFile = File(...),
        pixel_spacing_mm: float | None = Form(default=None),
    ) -> Module2Response:
        filename = file.filename or "upload"
        logger.info("POST /predict | filename=%s", filename)

        try:
            image_bytes = await file.read()
            processed, dicom_spacing = preprocess_panoramic(image_bytes)
            spacing = pixel_spacing_mm or dicom_spacing or get_default_pixel_spacing()
            return service.predict(processed, spacing, filename=filename)
        except ModelNotLoadedError as exc:
            logger.error("Model not loaded: %s", exc.model_name)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "models_not_loaded",
                    "model": exc.model_name,
                    "message": str(exc),
                    "hint": "Run: python training/train_all.py && python training/export_savedmodel.py",
                },
            ) from exc
        except ImageQualityError as exc:
            raise HTTPException(status_code=422, detail={"error": "image_quality_rejected", "message": str(exc)}) from exc
        except Exception as exc:
            logger.exception("Prediction failed for %s", filename)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return router
