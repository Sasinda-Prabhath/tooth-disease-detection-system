from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Point = tuple[float, float]
FDI = [str(q * 10 + t) for q in range(1, 5) for t in range(1, 9)]
DISCLAIMER = 'AI-assisted research output only. It is not a clinical diagnosis and requires dentist review.'

class Tooth(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    fdi: str
    box_xyxy: tuple[float, float, float, float]
    polygon: list[Point]
    confidence: float = Field(ge=0, le=1)
    status: Literal['detected', 'uncertain'] = 'detected'
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def valid_geometry(self):
        x1, y1, x2, y2 = self.box_xyxy
        if self.fdi not in FDI or min(x1, y1) < 0 or x2 <= x1 or y2 <= y1 or len(self.polygon) < 3:
            raise ValueError('Invalid FDI number, box or tooth polygon.')
        if any(min(point) < 0 for point in self.polygon):
            raise ValueError('Tooth coordinates must be non-negative.')
        return self

class ImpactionResult(BaseModel):
    fdi: str
    impacted: bool | None = None
    impaction_confidence: float | None = None
    angulation: str | None = None
    angulation_confidence: float | None = None
    gradcam_url: str | None = None
    reason: str | None = None

class Site(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    side: Literal['mesial', 'distal']
    cej_point: Point | None = None
    crest_point: Point | None = None
    apex_point: Point | None = None
    bone_loss_mm: float | None = None

class BoneResult(BaseModel):
    fdi: str
    assessable: bool = False
    sites: list[Site] = Field(default_factory=list)
    mean_bone_loss_mm: float | None = None
    severity: str = 'Not assessed'
    reason: str | None = None

class WorkflowStage(BaseModel):
    id: Literal['upload', 'preprocessing', 'fdi', 'impaction', 'bone_loss', 'report']
    label: str
    status: Literal['pending', 'running', 'completed', 'partial', 'blocked', 'failed'] = 'pending'
    detail: str | None = None
    image_url: str | None = None

class Prediction(BaseModel):
    case_id: str
    model_status: Literal['ready', 'degraded']
    inference_mode: str = 'research_prototype'
    raw_image_url: str | None = None
    processed_image_url: str | None = None
    fdi_preview_url: str | None = None
    impaction_preview_url: str | None = None
    bone_loss_preview_url: str | None = None
    final_report_url: str | None = None
    final_image_url: str | None = None
    image_width: int
    image_height: int
    pixel_spacing_mm: tuple[float, float] | None = None
    calibration_source: str | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    workflow: list[WorkflowStage] = Field(default_factory=list)
    teeth: list[Tooth] = Field(default_factory=list)
    third_molars: list[ImpactionResult] = Field(default_factory=list)
    bone_loss_results: list[BoneResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
