"""Proxy routes to Module 1 service."""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter(prefix="/module1", tags=["module1"])
MODULE1_URL = os.getenv("MODULE1_URL", "http://module1-service:8001")


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    pixel_spacing_mm: float = Form(default=0.1),
):
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        data = {"pixel_spacing_mm": str(pixel_spacing_mm)}
        resp = await client.post(f"{MODULE1_URL}/predict", files=files, data=data)
        resp.raise_for_status()
        return resp.json()


@router.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{MODULE1_URL}/health")
        resp.raise_for_status()
        return resp.json()
