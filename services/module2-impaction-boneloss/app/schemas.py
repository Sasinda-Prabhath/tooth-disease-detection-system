from pydantic import BaseModel, Field

from dental_common.schemas_base import BaseHealthResponse, BoundingBox


ANGULATION_LABELS = ["Mesioangular", "Vertical", "Horizontal", "Distoangular"]
SEVERITY_LABELS = ["Normal", "Mild", "Moderate", "Severe"]


class BoneLossResult(BaseModel):
    bone_loss_mm: float
    severity: str
    cej_points: list[list[float]] = Field(default_factory=list)
    crest_points: list[list[float]] = Field(default_factory=list)


class AngulationResult(BaseModel):
    label: str
    confidence: float
    is_impacted: bool = True


class ToothResult(BaseModel):
    fdi_number: int
    bbox: BoundingBox
    is_third_molar: bool = False
    angulation: AngulationResult | None = None
    bone_loss: BoneLossResult | None = None


class Module2Response(BaseModel):
    teeth: list[ToothResult]
    image_width: int
    image_height: int
    pixel_spacing_mm: float
    model_status: dict[str, bool] = Field(default_factory=dict)


class Module2HealthResponse(BaseHealthResponse):
    models_loaded: dict[str, bool] = Field(default_factory=dict)
