"""Placeholder proxy for Module 4 — Gingivitis/Stain (Priyawantha)."""

from fastapi import APIRouter

router = APIRouter(prefix="/module4", tags=["module4"])


@router.get("/health")
async def health():
    return {"status": "not_implemented", "service": "module4-gingivitis-stain"}
