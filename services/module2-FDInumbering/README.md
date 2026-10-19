# Dental FDI review

For clean-image placement, drawing tooth boxes, X/Y/red reference lines,
third-molar axes and six-point bone-model training, see
[dataset and measurement flow](docs/DATASET_AND_MEASUREMENT_FLOW.md).

For fixed numbering across missing teeth and step-by-step dataset preparation,
see [the training guide](docs/TRAINING_GUIDE.md). The app separates detected,
confirmed missing and unresolved FDI codes; missing 16/24/33 never shifts neighbors.

A Streamlit application for reviewing panoramic dental X-rays, FDI numbering,
third-molar axes, and downloadable single/bulk prediction overlays.

## Start

```powershell
python -m pip install -e .
python -m streamlit run app.py
```

The default is the built **YOLO26 + EfficientNet-B0 + ResNet50** bundle at
`runs/ensemble/yolo26_effnet_resnet50/model.fdi.zip`. All three actively contribute
on every detected tooth. The sidebar also retains detector-only and previous
two-model pipelines for comparison. Existing `FDI_MODEL_PATH` or
`FDI_HYBRID_PATH` environment overrides take precedence.

This is one packaged inference pipeline with three networks, not a jointly
trained single neural network. YOLO locates teeth; both classifiers process each
tooth crop and position. Their scores combine with YOLO evidence to select the
final FDI code and score. See [three-model design](docs/THREE_MODEL.md).

## Project structure

```text
app.py                         Streamlit launcher
pyproject.toml                 Package and dependencies
dental_fdi/
  paths.py                     Central project paths
  analysis/                    Angulation, FDI, periodontal and pose logic
  inference/                   YOLO, two-model and three-model inference
  imaging/                     X-ray preprocessing
  reporting/                   Shared overlays, CSV/JSON and ZIP exports
  ui/                          Streamlit screens and assets
  training/                    Detector/classifier training, bundle and evaluation
  data/                        Dataset validation
configs/                       Detection and pose dataset configurations
tests/                         Automated tests
tools/                         Local packaging, notebook and browser helpers
  legacy/                      Inactive original scripts retained as text
docs/                          Architecture, methodology and audit notes
notebooks/                     Updated Colab notebook
reports/                       Evaluations, logs and example outputs
models/pretrained/             Generic pretrained checkpoints
dataset/                       Existing local images and labels
runs/                          Existing training runs and model bundles
samples/                       Synthetic demonstration images
artifacts/                     Local training archives
```

## Train and evaluate

```powershell
python -m dental_fdi.training.train --check-only
python -m dental_fdi.training.train --task detect --epochs 100 --name new_detector
# Both classifier branches and their fusion bundle, using a dental detector:
python -m dental_fdi.training.train_ensemble --epochs 100
# Or combine the existing trained branches without retraining:
python -m dental_fdi.training.build_ensemble
python -m dental_fdi.training.evaluate_hybrid --hybrid runs/ensemble/yolo26_effnet_resnet50/model.fdi.zip --output reports/evaluation/three_model_comparison.json
python -m unittest discover -s tests
```

The classifier encoders are frozen ImageNet feature extractors; their dental FDI
heads were trained on 24 local OPGs. `train_ensemble` trains both heads and
packages them with the supplied detector. To use a newly trained detector, pass
`--detector path/to/best.pt`. This does not jointly train the three networks.
Patient-disjoint training/validation/test splits and expert labels are required.
A box classifier cannot recover a tooth that YOLO never proposed. Angulation
needs separately reviewed root/crown landmarks.

## Bulk uploads and downloads

Open **Bulk predictions**, select PNG/JPEG/TIFF/BMP files and click **Run batch
predictions**. Sidebar model, confidence, resolution and display settings apply
to every image. Check orientation and batch image adjustment; TIFF uses page 1.
Preview any overlay, download its PNG, or download all overlays and reports in
one verified ZIP. Failed files are recorded without stopping other files.
Duplicate filenames receive unique numbered outputs.

The ZIP includes `overlays/*.png`, `summary.csv`, `predictions.csv` and
`predictions.json`. CSV includes each model's top FDI and score. JSON additionally
includes all three evidence vectors, fusion weights, final labels and image
coordinates. The direct ZIP link under **ZIP download canceled or not starting?**
bypasses the media endpoint if the normal download is canceled by the browser.

`tooth_status.csv` lists all 32 permanent FDI codes per successfully processed
image. Confirm missing teeth separately for each previewed image; the overlay,
CSV and JSON update together. Undetected teeth remain unresolved until reviewed.

## Current accuracy limitation

On only 5 development validation images, at score threshold 0.35 and IoU >= 0.5,
the three-model pipeline scored precision **69.5%**, recall **37.3%**, and F1
**48.5%**. YOLO alone scored **64.6%**, **54.9%**, and **59.4%** respectively.
The conservative fused score lowers recall at this threshold. The ensemble
executes correctly but has not improved overall F1. Score thresholds are not
interchangeable across models and are not accuracy percentages. No independent
test set or clinical validation is available.

See [rebuild audit](docs/REBUILD_AUDIT.md), [previous two-model experiments](docs/HYBRID_MODEL.md)
and [angle workflow](docs/AUTOMATIC_ANGLES.md).

## YOLO26 bone-level measurements

See [the bone-level setup and annotation guide](docs/YOLO26_BONE_LOSS.md) for the six-landmark YOLO26-pose adaptation, patient-disjoint dataset checks, training, held-out evaluation and model activation. Expert CEJ/crest/apex annotations are required; tooth boxes alone cannot train bone measurements.
