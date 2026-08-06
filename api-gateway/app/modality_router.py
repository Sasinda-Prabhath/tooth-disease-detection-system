"""Route uploads to the correct microservice based on image type."""

from __future__ import annotations


def detect_modality(filename: str, content_type: str | None) -> str:
    """
    Simple modality router — X-ray vs clinical photo.
    Panoramic OPG / DICOM -> module2. Extend for other modules later.
    """
    name = (filename or "").lower()
    ctype = (content_type or "").lower()

    if name.endswith((".dcm", ".dicom")) or "dicom" in ctype:
        return "xray"
    if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        return "xray"
    return "unknown"


def services_for_modality(modality: str) -> list[str]:
    if modality == "xray":
        return ["module2"]
    return []
