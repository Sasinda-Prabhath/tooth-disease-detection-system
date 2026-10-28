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

### YOLO dataset in Google Colab

For a Roboflow/Ultralytics ZIP such as `Caries.v12i.yolo26.zip`, use
`docs/module1_colab_yolo_training.ipynb`. It uploads and extracts the ZIP,
checks `data.yaml`, trains a YOLO detector on the Colab GPU, tests it on an
X-ray, and downloads `best.pt`.

To use an exported YOLO enamel-erosion detector, place it at:

```text
models_store/enamel_erosion_detector/best.pt
```

The filename `yolo26_enamel_erosion_best.pt` is also supported.
The newer `enamel_erosion_yolo26s.pt` checkpoint is preferred when present.

The service runs all installed enamel models on the complete clinical image and converts
the detected erosion area into the existing `none`, `mild`, `moderate`, or
`severe` response. When both checkpoints are present, both models run for
comparison; the YOLO result remains the primary `enamel_erosion` response.
Their individual results are returned in `model_results`, and all detector
boxes are returned in `annotations` for frontend overlays. A Faster R-CNN
Torchvision state dict can be placed beside the YOLO file as
`models_store/enamel_erosion_detector/faster_rcnn_dental_erosion.pth`.

Color intraoral photographs are passed to the detector without grayscale
panoramic-X-ray preprocessing so the color and texture features used during
YOLO training are preserved.

Install the root requirements, then start the service on port 8001. The
existing `/predict` endpoint loads these files automatically.

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
- models_store/enamel_erosion_detector

## Endpoints

- GET /health
- POST /predict

`POST /predict` also reports the two caries pipeline stages (tooth detector and
caries classifier) in `model_results`, alongside the two enamel model results
when the corresponding checkpoints are installed. Both
`enamel_erosion_yolo26s.pt` and `yolo26_enamel_erosion_best.pt` are reported
separately in the frontend comparison.
