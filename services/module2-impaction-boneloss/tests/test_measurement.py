import pytest
import numpy as np
from app.inference.measurement import measure_bone_loss


def test_measure_bone_loss_normal():
    cej = [(100, 200), (150, 200)]
    crest = [(100, 190), (150, 190)]
    result = measure_bone_loss(cej, crest, pixel_spacing_mm=0.1)
    assert result["bone_loss_mm"] == 1.0
    assert result["severity"] == "Normal"


def test_measure_bone_loss_severe():
    cej = [(100, 300), (150, 300)]
    crest = [(100, 200), (150, 200)]
    result = measure_bone_loss(cej, crest, pixel_spacing_mm=0.1)
    assert result["severity"] == "Severe"
