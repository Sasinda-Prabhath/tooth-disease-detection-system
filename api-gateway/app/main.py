import os

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

MODULE2_URL = os.getenv("MODULE2_URL", "http://module2-service:8002")

app = FastAPI(title="Dental Disease Detection API Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "api-gateway"}


@app.post("/module2/predict")
async def module2_predict(
    file: UploadFile = File(...),
    pixel_spacing_mm: float = Form(default=0.1),
):
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        data = {"pixel_spacing_mm": str(pixel_spacing_mm)}
        resp = await client.post(f"{MODULE2_URL}/predict", files=files, data=data)
        resp.raise_for_status()
        return resp.json()
