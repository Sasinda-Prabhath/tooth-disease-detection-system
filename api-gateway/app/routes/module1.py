"""Placeholder proxy for Module 1 — Caries/Enamel (Prabhath)."""

from fastapi import APIRouter

router = APIRouter(prefix="/module1", tags=["module1"])


@router.get("/health")
async def health():
    return {"status": "not_implemented", "service": "module1-caries-enamel"}
