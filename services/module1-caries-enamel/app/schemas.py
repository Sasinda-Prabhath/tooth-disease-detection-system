from pydantic import BaseModel, Field

from dental_common.schemas_base import BaseHealthResponse, BoundingBox


CARIES_LABELS = ["none", "incipient", "moderate", "severe"]
ENAMEL_EROSION_LABELS = ["none", "mild", "moderate", "severe"]


class CariesResult(BaseModel):
    label: str
    confidence: float


class EnamelErosionResult(BaseModel):
    label: str
    confidence: float
    affected_surface_pct: float = 0.0


class ToothResult(BaseModel):
    fdi_number: int
    bbox: BoundingBox
    yolo_confidence: float | None = None
    caries: CariesResult | None = None
    enamel_erosion: EnamelErosionResult | None = None


class Module1Response(BaseModel):
    teeth: list[ToothResult]
    enamel_erosion: EnamelErosionResult | None = None
    image_width: int
    image_height: int
    pixel_spacing_mm: float
    model_status: dict[str, bool] = Field(default_factory=dict)
    model_results: dict[str, dict] = Field(default_factory=dict)
    annotations: list[dict] = Field(default_factory=list)


class Module1HealthResponse(BaseHealthResponse):
    models_loaded: dict[str, bool] = Field(default_factory=dict)
