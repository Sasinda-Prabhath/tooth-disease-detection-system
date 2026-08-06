from __future__ import annotations

import numpy as np

from app.schemas import SEVERITY_LABELS


def measure_bone_loss(
    cej_points: list[tuple[float, float]],
    crest_points: list[tuple[float, float]],
    pixel_spacing_mm: float,
) -> dict:
    """
    Measure vertical bone loss between CEJ and alveolar crest landmarks.
    Uses mesial and distal point pairs; returns average distance in mm.
    """
    if not cej_points or not crest_points:
        return {"bone_loss_mm": 0.0, "severity": "Normal"}

    distances_px = []
    for cej, crest in zip(cej_points, crest_points):
        distances_px.append(float(np.linalg.norm(np.array(cej) - np.array(crest))))

    distance_mm = float(np.mean(distances_px) * pixel_spacing_mm)

    if distance_mm < 2.0:
        severity = "Normal"
    elif distance_mm < 4.0:
        severity = "Mild"
    elif distance_mm < 6.0:
        severity = "Moderate"
    else:
        severity = "Severe"

    return {
        "bone_loss_mm": round(distance_mm, 2),
        "severity": severity,
        "cej_points": [[float(x), float(y)] for x, y in cej_points],
        "crest_points": [[float(x), float(y)] for x, y in crest_points],
    }


def severity_to_index(severity: str) -> int:
    try:
        return SEVERITY_LABELS.index(severity)
    except ValueError:
        return 0
