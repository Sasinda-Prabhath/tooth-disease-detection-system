> Historical two-model experiments. The current three-model default is described in [THREE_MODEL.md](THREE_MODEL.md).

# YOLO26 + EfficientNet / ResNet50

The app can load a single `.fdi.zip` model artifact containing the existing
32-class dental YOLO26 detector and a trained tooth classifier. One prediction
call runs both stages. This is a two-stage hybrid, not a fused backbone or an
end-to-end jointly trained neural network.

YOLO26 supplies tooth boxes and initial FDI classes. The classifier sees each
RGB tooth crop with 15% context padding, plus normalized box coordinates, to
review the FDI label. It supports EfficientNet-B0 and ResNet50. Box context
matters because a tooth crop alone may not establish patient side or quadrant.
The classifier cannot detect teeth that YOLO never proposed, measure molar
root/crown axes, or establish impaction.

The classifier changes an FDI label only at probability >= 0.85 and a margin
>= 0.25 over the original FDI class probability. These are conservative starting
rules, not clinically calibrated thresholds. A changed label's score is the
detector score multiplied by the classifier probability. An unchanged label
keeps its detector score. No confidence score is increased. Final postprocessing
keeps the strongest detection per FDI, as in detector-only mode.

## Train a candidate

```powershell
python -m dental_fdi.training.train_hybrid --architecture efficientnet_b0 --epochs 100
python -m dental_fdi.training.train_hybrid --architecture resnet50 --epochs 100
```

The default detector is `runs/detect/train-2/weights/best.pt`. Override it using
`--detector path/to/dental_yolo26.pt`. This must already be a trained 32-class
dental detector. Use the existing `dental_fdi/training/train.py` to fine-tune detection separately.

The classifier trainer uses the existing `dataset/images/{train,val}` and
`dataset/labels/{train,val}` YOLO annotations, with standard FDI class ordering.
It checks label structure and duplicate images. Patient-disjoint splitting and
expert annotation checks are still required. EXIF-rotated training images must
be normalized with their annotations before training.

For this small, CPU-only development setup, the ImageNet encoder stays frozen
and only the crop/context classification head is trained. This is a transfer
learning baseline, not a full fine-tune of EfficientNet or ResNet. Validation
selects the best head and early stopping limits unproductive epochs. More crops
from the same 24 OPGs do not become hundreds of independent patients.

Bundles appear in `runs/hybrid/<architecture>/candidate.fdi.zip`, alongside
training metrics. Bundles include a schema, FDI class order, weight checksums,
preprocessing version and training metrics. Loading uses no network downloads.

## Compare and use

```powershell
python -m dental_fdi.training.evaluate_hybrid --hybrid runs/hybrid/efficientnet_b0/candidate.fdi.zip --output reports/evaluation/hybrid_efficientnet_comparison.json
python -m dental_fdi.training.evaluate_hybrid --hybrid runs/hybrid/resnet50/candidate.fdi.zip --output reports/evaluation/hybrid_resnet50_comparison.json
```

The evaluator compares each bundle against its exact bundled YOLO detector at
the same confidence/resolution on raw RGB images. Matching requires the correct
FDI class and IoU >= 0.5. Reported precision, recall and F1 are fixed-operating-point
metrics, not mAP. Validation crop accuracy alone is not detection accuracy.
Use `--split test` with an independent annotated test set for a final assessment.
If the app uses enhancement, evaluate that preprocessing separately before use.

Restart or rerun Streamlit and choose a candidate in the sidebar **Prediction
model** selector. Both single-image and bulk prediction/export use the chosen
model. The detector remains the default. To configure a hybrid as default:

```powershell
$env:FDI_HYBRID_PATH = 'C:/path/to/candidate.fdi.zip'
python -m streamlit run app.py
```

Use only locally trained or trusted model bundles, as their YOLO checkpoints
use the same checkpoint loader as ordinary Ultralytics models.

## Architecture choice

EfficientNet-B0 is the lower-cost CPU starting point; ResNet50 is a larger
comparison encoder. Neither is known to be better for this dataset before
evaluation. YOLO26 already provides a detection backbone. Replacing it with an
EfficientNet backbone is a separate architecture experiment requiring detector
training and suitable multi-scale feature connections; adding a crop classifier
does not perform that replacement. Prioritize label quality, patient diversity
and detection recall before increasing model complexity.

Official references:
- https://docs.ultralytics.com/models/yolo26/
- https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
- https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html

## Local development results

Both candidates were trained on the existing 24 training OPGs (635 annotated
tooth crops) and selected using 5 validation OPGs (153 annotated tooth crops).
Only their classification heads were trained; the encoders and detector stayed
frozen. EfficientNet used crop batch size 32 and ResNet50 used 16, so these are
initial development runs rather than a controlled architecture benchmark.

At confidence 0.35, resolution 1024 and correct-FDI IoU >= 0.5 matching:

| Pipeline | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Existing YOLO26 | 64.6% | 54.9% | 59.4% |
| YOLO26 + EfficientNet-B0 | 63.8% | 54.2% | 58.7% |
| YOLO26 + ResNet50 | 63.6% | 53.6% | 58.2% |

Neither hybrid improved this development set. The app therefore keeps the
detector as default and displays the saved comparison outcome for each hybrid.
Both pipelines execute; neither is established as a more accurate replacement.
The encoder choice cannot be settled on five development images. No independent
test set is currently available, and no clinical accuracy claim follows.

The complete results are `reports/evaluation/hybrid_efficientnet_comparison.json` and
`reports/evaluation/hybrid_resnet50_comparison.json`. The evaluation is threshold-specific and
differs from Ultralytics' mAP and best-F1 operating-point summaries.
