"""Proxy routes to Module 2 service."""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/module2", tags=["module2"])
MODULE2_URL = os.getenv("MODULE2_URL", "http://module2-service:8002")


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    pixel_spacing_mm: float | None = Form(default=None, gt=0),
):
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        data = {"pixel_spacing_mm": str(pixel_spacing_mm)} if pixel_spacing_mm is not None else {}
        resp = await client.post(f"{MODULE2_URL}/predict", files=files, data=data)
        if resp.is_error:
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        resp.raise_for_status()
        return resp.json()


@router.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{MODULE2_URL}/health")
        resp.raise_for_status()
        return resp.json()
