# Module 2 — System Architecture

**Author:** Lakshitha R.M.S.K (IT23222618)

## Recommended pattern

| Layer | Pattern | This repo |
|-------|---------|-----------|
| **Backend (FastAPI)** | **MVC** | **Controller** → **Service** → **Inference (Model)** |
| **Frontend (React)** | **MVVM** | **View** (components) → **ViewModel** (`useModule2Prediction`) → **API client** |

Use **MVC on the backend** and **MVVM on the frontend**. Do not mix MVC inside React (no fat controllers in components).

---

## Backend MVC (Module 2)

```
frontend POST /predict
    │
    ▼
app/controllers/predict_controller.py   ← HTTP, validation, errors
    │
    ▼
app/services/prediction_service.py      ← pipeline orchestration + logging
    │
    ├── app/inference/tooth_segmenter.py      Stage 0: all teeth (AKUDENTAL-style)
    ├── app/inference/detector.py             Stage A: third molars FDI 18/28/38/48
    ├── app/inference/angulation.py           Stage B: angulation class
    ├── app/inference/bone_landmark.py        Stage C: CEJ + crest
    └── app/inference/measurement.py          mm bone loss + severity
```

### Configuration

- `app/config.py` — `MODELS_DIR`, `REQUIRE_TRAINED_MODELS` (default **on**), `LOG_LEVEL`
- With `REQUIRE_TRAINED_MODELS=1`, the API returns **503** if any model is missing. **No heuristic/mock predictions.**

### Logging

Backend logs each upload and stage to stdout:

```
module2.controller | POST /predict | filename=...
module2.prediction | Starting prediction | file=... | mode=ml
module2.prediction | Stage 0: tooth instance segmentation
...
```

Run uvicorn in a terminal and watch logs while uploading from the UI.

---

## Frontend MVVM

```
App.jsx (View)
  └── useModule2Prediction.js (ViewModel: state + predict())
        └── api/client.js (Model: HTTP to backend)
  └── AnnotationOverlay.jsx — colored instance masks + CEJ/crest
  └── ReportView.jsx — third molar report
```

Upload always calls **`POST http://localhost:8002/predict`**. The browser does not run inference.

---

## Four trained models (required for production)

| Model | Role | Train script | Export path |
|-------|------|--------------|-------------|
| `tooth_instance_segmenter` | Tooth-by-tooth colored masks (like AKUDENTAL) | `training/train_tooth_instance.py` | `models_store/tooth_instance_segmenter/` |
| `molar_detector` | Third molar boxes | `training/train_detector.py` | `models_store/molar_detector/` |
| `angulation_classifier` | Mesioangular / Vertical / Horizontal / Distoangular | `training/train_angulation.py` | `models_store/angulation_classifier/` |
| `bone_landmark_unet` | CEJ + alveolar crest lines | `training/train_segmentation.py` | `models_store/bone_landmark_unet/` |

Full pipeline:

```powershell
cd services\module2-impaction-boneloss
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

# Bootstrap synthetic data (smoke test only)
python training\generate_sample_data.py --num-images 40

# Train all + export (quick mode ~20 min)
$env:MODULE2_QUICK_TRAIN = "1"
python training\train_all.py --skip-data --quick
```

For **real OPGs**, replace synthetic COCO with expert annotations (see below), then train **without** `--quick`.

---

## Real data annotation (AKUDENTAL-style)

Your **161** images in `training/data/raw/` are **not** in `coco_annotations.json` until you annotate them.

### COCO format (`training/data/annotations/coco_annotations.json`)

Per tooth instance:

```json
{
  "bbox": [x, y, width, height],
  "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]],
  "attributes": {
    "fdi_number": 37,
    "angulation": "Mesioangular",
    "impacted": true
  }
}
```

- **All teeth (11–48):** `segmentation` polygon + `fdi_number` for instance segmenter.
- **Third molars only:** add `angulation` and `impacted`.

### Bone landmarks (Stage C)

Per third molar crop: `training/data/processed/segmentation_masks/{image_stem}_fdi{nn}.npy`  
Shape `H × W × 2` — channel 0 = CEJ, channel 1 = crest.

This folder is **gitignored** (large binary masks).

### Tools

- [CVAT](https://www.cvat.ai/) or Label Studio — export COCO with segmentation
- [AKUDENTAL dataset](https://bmcoralhealth.biomedcentral.com/) — reference for mask quality
- Clinical supervisor — validate angulation and bone-loss mm

---

## Accuracy expectations

**100% accuracy is not achievable** in medical imaging. Target project metrics (`training/config.yaml`):

| Metric | Target |
|--------|--------|
| Detection mAP@0.5 | ≥ 0.90 |
| Angulation macro-F1 | ≥ 0.85 |
| Segmentation Dice | ≥ 0.80 |
| Bone loss MAE | ≤ 1.0 mm |
| Severity Cohen's κ | ≥ 0.80 |

Report these on a **held-out expert-annotated test set**, not training data.

---

## Recommendations

1. **Annotate real OPGs first** — synthetic data only proves the pipeline; it will not generalize to your panoramics.
2. **Use AKUDENTAL or similar** for pre-training instance segmentation, then fine-tune on your 161 images.
3. **Upgrade detector** to full YOLOv8 (`training/models/detector_model.py`) when COCO has many real boxes per image.
4. **DICOM pixel spacing** when available; document manual mm/px for JPG uploads.
5. **Keep `REQUIRE_TRAINED_MODELS=1`** in demos and submission so reviewers never see heuristic 7.86 mm / 65% placeholders.
6. **Split data** 80/10/10 train/val/test with patient-level splits (no leakage).
7. **Modules 1, 3, 4** — separate services; copy this MVC + MVVM layout per module.

---

## Run locally

```powershell
# Terminal 1 — API
cd services\module2-impaction-boneloss
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
$env:REQUIRE_TRAINED_MODELS = "1"
uvicorn app.main:app --port 8002 --reload

# Terminal 2 — UI
cd frontend
$env:VITE_API_BASE_URL = "http://localhost:8002"
npm run dev
```

Open http://localhost:3000, upload an X-ray, confirm backend logs and colored tooth overlays when all four models are loaded.
