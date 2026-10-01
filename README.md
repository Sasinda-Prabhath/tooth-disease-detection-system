# Tooth Disease Detection System

Module 2 implements a third-molar **research** pipeline and four-stage React + TypeScript UI:

**Original OPG → privacy-masked processed OPG → FDI boxes/contours → final analysis.**

The dental stack is PyTorch: YOLO26l-seg FDI detection, EfficientNetV2-S + geometry impaction/angulation, and nnU-Net v2 2D CEJ/crest segmentation. Privacy masking uses PP-OCRv6 through PaddleOCR. Inference consumes clean pixels and structured coordinates, never rendered annotations.

No trained dental models, privacy model exports, patient data or clinical accuracy results are included. Missing/incompatible models report `degraded` and do not produce substitute predictions. The bone landmark conversion is an unvalidated research heuristic; dentist review is required.

## Start locally (Python 3.11)

```powershell
.\scripts\setup-module2.ps1
.\scripts\start-module2.ps1
```

Open http://localhost:3000. API: http://localhost:8002/module2/health. Add `-ML` to setup when installing model runtimes. Provision trained artifacts and restart to enable stages.

```sh
docker compose -f docker-compose.module2-only.yml up --build
```

The Compose stack includes PostgreSQL and private local storage. See the module guide for ML/GPU images, S3 configuration, session access and retention.

- [Module 2 implementation and limitations](services/module2-impaction-boneloss/README.md)
- [Patient-separated annotation, training and evaluation](services/module2-impaction-boneloss/training/HYBRID_TRAINING.md)
- [Research methodology](docs/module2_methodology.md)

The existing API gateway forwards Module 2 routes. Other service directories remain separate project modules. Synthetic test fixtures establish software behavior only, not diagnostic performance.
