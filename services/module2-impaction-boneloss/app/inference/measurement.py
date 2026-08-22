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


def offset_landmarks_to_image(
    points: list[tuple[float, float]],
    origin_x: float,
    origin_y: float,
) -> list[tuple[float, float]]:
    return [(origin_x + x, origin_y + y) for x, y in points]


def compute_periodontal_stage(
    bone_loss_mm_values: list[float],
    teeth_detected: int,
    expected_teeth: int = 28,
) -> int:
    """
    Map arch-wide bone loss to periodontal stage (I–IV) for OPG summary overlay.
    Uses max interdental bone loss and missing-tooth ratio (clinical-style staging).
    """
    if not bone_loss_mm_values:
        return 1

    max_loss = max(bone_loss_mm_values)
    mean_loss = sum(bone_loss_mm_values) / len(bone_loss_mm_values)
    missing_ratio = max(0.0, 1.0 - teeth_detected / expected_teeth)

    if max_loss >= 6.0 or (missing_ratio >= 0.2 and mean_loss >= 4.0):
        return 4
    if max_loss >= 4.0 or mean_loss >= 3.5:
        return 3
    if max_loss >= 2.0 or mean_loss >= 1.5:
        return 2
    return 1
