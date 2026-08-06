# Tooth Disease Detection System

Microservices-based dental AI platform. (Impaction & Bone Loss)

## Repository Layout

```
tooth-disease-detection-system/
├── docker-compose.yml                 # Full stack (gateway + module2 + frontend + postgres)
├── docker-compose.module2-only.yml    # Module 2 solo dev/demo
├── frontend/                          # Shared React UI
├── api-gateway/                       # Routes to microservices
├── services/
│   └── module2-impaction-boneloss/    # ★ Your module (fully functional)
├── shared-lib/                        # Minimal shared utilities
├── infra/                             # Postgres init scripts
└── docs/                              # Methodology docs
```

## Module 2 Features

- Panoramic X-ray upload (PNG/JPG/DICOM)
- Third molar detection (FDI 18, 28, 38, 48)
- Angulation classification: Mesioangular, Vertical, Horizontal, Distoangular
- CEJ + alveolar crest landmark segmentation
- Bone loss measurement in mm with severity grading
- React UI with bounding boxes and landmark overlay (Konva.js)

---

## Quick Start (Recommended — Docker)

### Prerequisites
- Docker Desktop
- 8 GB+ RAM (TensorFlow container)

### Step 1 — Train models (first time only)

```powershell
cd services\module2-impaction-boneloss
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ..\..\shared-lib
pip install -r requirements.txt
python training\train_all.py
```

This will:
1. Generate 40 synthetic bootstrap images (replace with your real OPG data later)
2. Train detector, angulation classifier, and U-Net segmenter
3. Export TensorFlow SavedModels to `models_store/`

### Step 2 — Run Module 2 + Frontend

```powershell
cd ..\..
docker compose -f docker-compose.module2-only.yml up --build
```

### Step 3 — Test in browser

1. Open **http://localhost:3000**
2. Upload a panoramic X-ray (PNG/JPG) or use a synthetic image from `training/data/raw/`
3. Set **pixel spacing (mm/pixel)** — use DICOM value or `0.1` as fallback
4. View angulation badges, bone loss chart, and annotated overlay

API docs: **http://localhost:8002/docs**

---

## Local Development (Without Docker)

### Backend

```powershell
cd services\module2-impaction-boneloss
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

### Frontend

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL = "http://localhost:8002"
npm run dev
```

Open **http://localhost:3000**

---

## Training on Your Real Dataset

### 1. Add panoramic X-rays

Place OPG images in:
```
services/module2-impaction-boneloss/training/data/raw/
```

### 2. Annotate in COCO format

Edit or replace:
```
training/data/annotations/coco_annotations.json
```

Each third molar annotation needs:
```json
{
  "bbox": [x, y, width, height],
  "attributes": {
    "fdi_number": 38,
    "angulation": "Mesioangular",
    "impacted": true
  }
}
```

### 3. Add segmentation masks

Save `.npy` masks (H×W×2) in:
```
training/data/processed/segmentation_masks/
```
Channel 0 = CEJ line, Channel 1 = alveolar crest line.

### 4. Train each stage

```powershell
python training\train_detector.py
python training\train_angulation.py
python training\train_segmentation.py
python training\export_savedmodel.py
```

Or run all at once: `python training\train_all.py`

### 5. Evaluate

```powershell
python training\evaluate.py
```

Target metrics (see `training/config.yaml`):
| Metric | Target |
|--------|--------|
| mAP@0.5 | > 0.90 |
| Angulation F1 | > 0.85 |
| Dice | > 0.80 |
| Bone loss MAE | < 1.0 mm |
| Cohen's Kappa | > 0.80 |

---

## API Endpoints (Module 2)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service + model load status |
| POST | `/predict` | Upload X-ray, get full analysis |

Example (curl):
```powershell
curl -X POST "http://localhost:8002/predict" `
  -F "file=@panoramic.png" `
  -F "pixel_spacing_mm=0.1"
```

---

## Makefile Commands

```bash
make module2          # Docker: module2 + frontend
make train-module2    # Train all models
make export-module2   # Export SavedModels
make test-module2     # Run pytest
make frontend-dev     # Local Vite dev server
```

---

## Notes for Report

- **Pixel calibration**: DICOM `PixelSpacing` is auto-read when available. For PNG datasets, document your fallback calibration method.
- **Severity thresholds** in `app/inference/measurement.py` are illustrative — confirm with Dr. Savith Siriwardane before final submission.
- **Heuristic fallback**: If `models_store/` is empty, the API still returns results using rule-based fallbacks (shown in UI as "Heuristic fallback").

---

## Team Modules (Coming Soon)

| Module | Owner | Port |
|--------|-------|------|
| Module 1 — Caries/Enamel | Prabhath | 8001 |
| Module 2 — Impaction/Bone Loss | Lakshitha | 8002 |
| Module 3 — Decay/Fracture | Jayasundara | 8003 |
| Module 4 — Gingivitis/Stain | Priyawantha | 8004 |

Copy `services/module2-impaction-boneloss/` folder structure for other modules.
