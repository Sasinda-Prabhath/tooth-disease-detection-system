import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import module1, module2, module3, module4, unified_report

app = FastAPI(title="Dental Disease Detection API Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(module1.router)
app.include_router(module2.router)
app.include_router(module3.router)
app.include_router(module4.router)
app.include_router(unified_report.router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "api-gateway",
        "module2_url": os.getenv("MODULE2_URL", "http://module2-service:8002"),
    }
