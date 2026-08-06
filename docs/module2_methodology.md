# Module 2 — Impaction & Bone Loss Methodology

**Author:** Lakshitha R.M.S.K (IT23222618)

## Pipeline Stages

1. **Detection** — Locate third molars (FDI 18, 28, 38, 48) via object detection
2. **Angulation** — Classify Mesioangular / Vertical / Horizontal / Distoangular
3. **Bone loss** — Segment CEJ + alveolar crest, measure mm distance

## Calibration

Use DICOM `PixelSpacing` when available. For PNG/JPG datasets, set `pixel_spacing_mm` in the UI or use a reference scale (document limitation in report).

## Severity Thresholds (illustrative)

| mm | Grade |
|----|-------|
| < 2 | Normal |
| 2–4 | Mild |
| 4–6 | Moderate |
| ≥ 6 | Severe |

Confirm final thresholds with clinical supervisor before submission.
