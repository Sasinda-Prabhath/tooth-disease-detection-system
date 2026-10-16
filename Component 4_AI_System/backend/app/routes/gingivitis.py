from __future__ import annotations

import traceback

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas.gingivitis import HealthResponse, PredictionResponse
from app.services.gingivitis_service import PredictionError


router = APIRouter(tags=["gingivitis"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict[str, str | None]:
    return request.app.state.gingivitis_service.health()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.45),
    iou_threshold: float = Form(0.45),
    max_detections: int = Form(12),
) -> dict:
    print("[/predict] request received", flush=True)
    print(f"[/predict] uploaded filename: {image.filename}", flush=True)
    print("Received confidence:", confidence_threshold, flush=True)
    print("Received IoU:", iou_threshold, flush=True)
    print("Received max detections:", max_detections, flush=True)

    if not image.filename:
        raise HTTPException(status_code=400, detail="Missing image filename.")

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")
        print("[/predict] prediction starts", flush=True)
        return request.app.state.gingivitis_service.predict(
            image_bytes=image_bytes,
            filename=image.filename,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
    except ValueError as exc:
        print("[/predict] validation error", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PredictionError as exc:
        print("[/predict] prediction error", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        print("[/predict] unexpected exception", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
