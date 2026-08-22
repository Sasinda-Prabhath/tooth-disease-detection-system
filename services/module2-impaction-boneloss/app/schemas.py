from pydantic import BaseModel, ConfigDict, Field

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


class ToothSegmentationResult(BaseModel):
    fdi_number: int
    label: str
    confidence: float
    bbox: BoundingBox
    contour: list[list[float]] = Field(default_factory=list)
    color_rgb: list[int] = Field(default_factory=lambda: [200, 200, 200])


class ToothResult(BaseModel):
    fdi_number: int
    bbox: BoundingBox
    is_third_molar: bool = True
    detection_confidence: float | None = None
    angulation: AngulationResult | None = None
    bone_loss: BoneLossResult | None = None
    segmentation: ToothSegmentationResult | None = None


class Module2Response(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    teeth: list[ToothResult]
    all_teeth_segments: list[ToothSegmentationResult] = Field(default_factory=list)
    image_width: int
    image_height: int
    pixel_spacing_mm: float
    model_status: dict[str, bool] = Field(default_factory=dict)
    inference_mode: str = "trained_yolov8"
    periodontal_stage: int | None = Field(default=None, ge=1, le=4)
    mean_bone_loss_mm: float | None = None


class Module2HealthResponse(BaseHealthResponse):
    models_loaded: dict[str, bool] = Field(default_factory=dict)
