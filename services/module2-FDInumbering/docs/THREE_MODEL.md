# Three active models, one prediction pipeline

`runs/ensemble/yolo26_effnet_resnet50/model.fdi.zip` contains the trained dental
YOLO26 detector, EfficientNet-B0 dental classifier, and ResNet50 dental classifier.
The bundle includes FDI order, preprocessing version, component checksums,
training histories and fusion settings. Loading requires no model downloads.

Every inference call follows this sequence:

1. YOLO26 processes the full image and proposes tooth boxes and FDI scores.
2. Each proposal becomes an RGB crop with 15% context padding. Both classifiers
   receive the same 224x224 crop and normalized box coordinates.
3. Both networks execute on every proposal. Their complete 32-class softmax
   vectors contribute to the final FDI class; no disagreement gate bypasses them.
4. The final label and score use all three components. Postprocessing applies
   the requested score threshold and keeps the strongest detection per FDI.

This is decision-level fusion, not a jointly optimized backbone or a single
exported ONNX graph. YOLO alone supplies geometry; the two classifiers refine
identity and score. They cannot create boxes for teeth YOLO never proposed.

## Fusion definition

Let `s` be YOLO's winning score and `c` its winning FDI index. Ultralytics' public
box result does not expose the complete class distribution. The explicit YOLO
evidence approximation is `Y = s * one_hot(c) + (1-s) * uniform(32)`.
The uniform prior keeps the detector's winning class highest even for low scores;
this does not invent raw class logits that the detector did not expose.

For classifier vectors `E` and `R`, the final class is the argmax of:

```text
F = 0.50 * Y + 0.25 * E + 0.25 * R
final_class = argmax(F)
final_score = s * (0.50 + 0.25 * E[final_class] + 0.25 * R[final_class])
```

Both classifiers therefore influence scores even when the final label stays
unchanged. Every weight must be positive; disabling a component raises an error.
The final score cannot exceed the detector score. This conservative rule can
lower recall at a fixed threshold. The fused vector and score are uncalibrated
evidence, not independently established probabilities of correctness.

The starting weights were not optimized on the five validation images. Tune
weights/thresholds on a larger patient-disjoint development set, then freeze them
before evaluation on an independent test set. Comparing at 35% does not imply
that 35% has the same operating meaning for every pipeline.

## Rebuild, train and evaluate

```powershell
# Reuse the two already trained classifiers; require identical detector bytes:
python -m dental_fdi.training.build_ensemble
# Train both classifier heads with matching settings, then create the bundle:
python -m dental_fdi.training.train_ensemble --epochs 100 --batch 32
# The detector can be fine-tuned separately with reviewed tooth boxes:
python -m dental_fdi.training.train --task detect --epochs 100 --name detector_candidate
# Use a particular dental detector for both classifier bundles:
python -m dental_fdi.training.train_ensemble --detector runs/detect/detector_candidate/weights/best.pt
# Evaluate the assembled pipeline against its exact bundled detector:
python -m dental_fdi.training.evaluate_hybrid --hybrid runs/ensemble/yolo26_effnet_resnet50/model.fdi.zip --output reports/evaluation/three_model_comparison.json
```

The provided bundle reuses the two existing trained classifier heads with frozen
ImageNet encoders. It does not imply a new joint training run. See the historical
two-model experiment notes for training limitations. A larger, expert-labeled
dataset is needed before changing to more elaborate training regimes.

## Inspect contributions

The single-image review includes **Contributions from all three models**.
Bulk `predictions.csv` reports the FDI and score from each classifier.
`predictions.json` contains `model_contributions` on every retained automatic
prediction: YOLO evidence, both classifier vectors, and each positive weight.
It also includes `fusion_scores`, `fusion_class_score`, detector identity and the
final decision. Manual box corrections are marked separately.

## Current development evaluation

Five images, 153 labeled teeth, raw RGB, resolution 1024, score threshold 0.35,
correct-FDI IoU matching >= 0.5:

| Pipeline | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| YOLO26 alone | 64.6% | 54.9% | 59.4% |
| YOLO26 + EfficientNet-B0 + ResNet50 | 69.5% | 37.3% | 48.5% |

These results establish execution, not improved accuracy. The ensemble is the
requested default, with a visible evaluation notice and detector-only comparison
available. No independent test set is available. See
`reports/evaluation/three_model_comparison.json` for exact counts and settings.
