# Training Module 2 with your OPG dataset

The production report uses trained models only. It does not invent third-molar boxes when YOLO cannot find a tooth.

## 1. Label your panoramic X-rays

Put de-identified OPG images in `training/data/raw/`. Annotate every visible third molar with a bounding box and its exact FDI number.

The detector has exactly four classes, in this fixed order:

| YOLO class | FDI tooth |
| --- | --- |
| 0 | 18 |
| 1 | 28 |
| 2 | 38 |
| 3 | 48 |

The supplied COCO file accepts annotations such as:

```json
{
  "image_id": 1,
  "bbox": [120, 210, 82, 115],
  "attributes": {
    "fdi_number": 48,
    "angulation": "Horizontal",
    "impacted": true
  }
}
```

Use your annotation tool's COCO export, then ensure each image entry has `id`, `file_name`, `width`, and `height`.

## 2. Build the YOLO dataset and train the detector

From `services/module2-impaction-boneloss` with the virtual environment active:

```powershell
python training\prepare_yolo_dataset.py
python training\train_detector.py --model yolov8s.pt --epochs 150 --imgsz 1024
```

For a CPU-only computer, add `--device cpu`; training will be much slower. The training script copies the selected weight file to:

```text
models_store/molar_detector/best.pt
```

Restart the API after training. `GET /health` must report `molar_detector_yolov8: true`.

## 3. Train the other report models

YOLO detects the teeth only. To produce a valid complete report, also provide:

- `angulation` and `impacted` attributes in the COCO annotations for each third molar crop.
- CEJ/crest masks for each labelled tooth: `training/data/processed/segmentation_masks/<image-or-tooth>.npy`, where channel 0 is CEJ and channel 1 is alveolar crest.
- Full-arch tooth masks when you want the coloured tooth visualization.

Then train and export the remaining models:

```powershell
python training\train_angulation.py
python training\train_segmentation.py
python training\train_tooth_instance.py
python training\export_savedmodel.py
```

Do not evaluate this as a medical device until you validate it on a held-out, expert-annotated test set. Pixel spacing must come from the DICOM file or a validated calibration method; `0.1 mm/px` is only a fallback value.
