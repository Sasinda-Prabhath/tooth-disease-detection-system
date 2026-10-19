# Automatic angle estimation (no landmark dataset)

The default app now runs an unsupervised image method immediately after tooth
detection. It thresholds each molar crop at three nearby intensity levels,
selects a connected component and fits its principal axis. It rejects weak
contrast, nearly round components, missing adjacent molars and axes that vary
by more than 15 degrees between segmentations.

The overlay shows amber third-molar axes, cyan second-molar axes and estimated
angles for 18/17, 28/27, 38/37 and 48/47 where both axes are available. It does
not derive an angle from the bounding rectangle. Tests verify that changing
pixels inside an identical box changes the estimated axis.

**These estimates are unsigned (0–90 degrees), unvalidated and may be wrong.**
A principal axis cannot identify which end is the crown. Therefore it cannot
resolve inverted teeth or produce a trustworthy signed Winter category without
root/crown review. Stable threshold results are a numerical consistency check,
not a clinical confidence interval or proof of correct segmentation.

Use **Review or correct axes for signed Winter classification** to inspect
prefilled proposals. Correct the landmarks, swap crown/root ends if needed,
and confirm them. The reviewed result uses the original article's signed ranges.
Teeth 18/28 remain outside the mandibular study's classification scope. Neither
this algorithm nor the current detection checkpoint determines impaction status.
It would require additional anatomic/clinical evidence or labeled training data.

The image method is a fallback, not a substitute for a validated landmark model.
It can fail when roots overlap bone or adjacent teeth, crown restorations dominate
intensity, teeth are nearly round in projection, or detection boxes omit anatomy.
Unavailable angles stay unavailable rather than being filled with zero.

## Fine-tuning

`dental_fdi/training/train.py` now starts detection fine-tuning from the installed dental checkpoint,
uses gentle augmentation without mosaic/mixup/flips, and saves separate candidate
weights. The local dataset contains 24 training and 5 validation images. Passing
structural checks does not establish anatomical label quality or patient separation.

```powershell
python -m dental_fdi.training.train --task detect --check-only
python -m dental_fdi.training.train --task detect --epochs 100 --device 0 --batch 4 --name fdi_candidate
python -m dental_fdi.training.evaluate_models --candidate runs/detect/fdi_candidate/weights/best.pt
```

`FDI_Finetuning_Colab.zip` contains the local dataset, current starting weights and
training code. Open `Dental_FDI_YOLO26_Training.ipynb` in a GPU Colab runtime and
upload the bundle there. No images or weights were uploaded by this work.
The notebook downloads the actual trainer output rather than a hardcoded run path.

If expert root/crown labels become available, `configs/pose_data.yaml` and the optional
pose trainer provide an eight-class, two-keypoint route. `dental_fdi/analysis/pose_pipeline.py`
rejects bounding-box-only labels and incompatible human-pose checkpoints.
Set `FDI_AXIS_MODEL_PATH` to the trained pose weights and include the generated
`axis_schema.json` beside them. The app then uses predicted landmarks instead
of intensity-based axes. Training labels must not be generated from these
unverified automatic estimates and treated as ground truth.

Sources:
- Study: https://pmc.ncbi.nlm.nih.gov/articles/PMC11607221/
- OpenCV component/shape operations: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
- Ultralytics pose format: https://docs.ultralytics.com/datasets/pose/

Tests: `python -m unittest test_dental_analysis test_automatic_angulation -v`
