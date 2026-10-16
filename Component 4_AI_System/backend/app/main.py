from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.gingivitis import router as gingivitis_router
from app.services.gingivitis_service import (
    GingivitisHybridModelService,
    ModelLoadError,
)


model_service = GingivitisHybridModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load_models()
    except ModelLoadError:
        pass
    yield


app = FastAPI(
    title="Oral AI System API",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.gingivitis_service = model_service

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gingivitis_router)
