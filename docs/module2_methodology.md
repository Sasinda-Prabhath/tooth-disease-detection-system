# Module 2 methodology

Scope: impacted third molars 18/28/38/48 and research CEJ-to-alveolar-crest distance, not full-mouth periodontal diagnosis.

1. Freeze patient-separated datasets before annotation/training; keep repeated scans in one partition.
2. Review privacy masking and preprocessing; block inference if OCR is unavailable or detects unresolved text within diagnostic anatomy.
3. Detect all visible permanent teeth with YOLO26l-seg using 32 FDI classes. Keep uncertain detections uncertain; do not invent missing teeth.
4. Extract clean third-molar plus adjacent-second-molar ROIs using coordinates.
5. EfficientNetV2-S image features plus geometric features feed impaction and angulation heads. Non-impacted results have null angulation.
6. Segment CEJ and crest using nnU-Net v2 2D. Dense expert masks are required in addition to validation landmarks. Per-surface centroid extraction is an unvalidated heuristic and abstains on ambiguous or substantially tilted/horizontal tooth axes.
7. Measure Euclidean distances using verified row and column spacing. Missing calibration means null millimetres. Panoramic projection distortion remains a measurement limitation. Never infer scale from a typical tooth size.
8. Keep severity `Not assessed`. No automatic periodontal stage or unvalidated severity thresholds.
9. Evaluate frozen models on unseen patients and request dentist review of failures, missing landmarks, uncertainty and calibration.

Required metrics: FDI mAP@0.5/precision/recall; impaction precision/recall/F1; angulation accuracy/macro F1/confusion matrix; CEJ/crest Dice; measurement MAE in mm; coverage/abstention. Severity kappa is only reported after dentist-approved grading definitions exist. Software unit tests and synthetic fixtures are not clinical evidence.

See [the full ordered protocol](../services/module2-impaction-boneloss/training/HYBRID_TRAINING.md) and [runtime limitations](../services/module2-impaction-boneloss/README.md).
