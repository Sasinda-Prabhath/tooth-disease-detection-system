from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class BaseHealthResponse(BaseModel):
    status: str = "ok"
    service: str


class BasePredictionRequest(BaseModel):
    pixel_spacing_mm: float = Field(
        default=0.1,
        gt=0,
        description="Millimeters per pixel (from DICOM PixelSpacing or calibration fallback).",
    )
