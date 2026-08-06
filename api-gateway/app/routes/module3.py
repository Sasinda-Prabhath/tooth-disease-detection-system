"""Placeholder proxy for Module 3 — Decay/Fracture (Jayasundara)."""

from fastapi import APIRouter

router = APIRouter(prefix="/module3", tags=["module3"])


@router.get("/health")
async def health():
    return {"status": "not_implemented", "service": "module3-decay-fracture"}
