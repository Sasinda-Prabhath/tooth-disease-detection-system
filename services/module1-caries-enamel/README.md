# Module 1 - Caries and Enamel Erosion

This service now includes a practical scaffold for:
- Tooth-region detection
- Caries severity classification
- Enamel erosion estimation

## Service

Run from this folder:

```powershell
$env:PYTHONPATH=(Get-Location).Path
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## Training Data Layout

- training/data/raw/
- training/data/annotations/coco_annotations.json
- training/data/processed/enamel_masks/*.npy

Expected COCO annotation attributes per instance:
- tooth_fdi
- caries_grade (none, incipient, moderate, severe)
- enamel_grade (none, mild, moderate, severe)

## Training

Quick bootstrap:

```powershell
$env:MODULE1_QUICK_TRAIN="1"
python training/train_all.py --quick
```

Full training:

```powershell
python training/train_all.py
```

## Exported Models

- models_store/caries_detector
- models_store/caries_classifier
- models_store/enamel_erosion_unet

## Endpoints

- GET /health
- POST /predict
