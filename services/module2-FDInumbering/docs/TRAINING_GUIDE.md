# Train correct FDI numbering when teeth are missing

## Expected result

Each visible permanent tooth has its own anatomical FDI code and a box covering
that tooth. The sequence is not a count of the visible teeth. For example,
if 16 is absent, 17 and 18 remain 17 and 18. Missing 24 does not turn 25 into 24;
missing 33 does not turn 34 into 33.

Patient right appears on image left in the standard display:

```text
Upper: 18 17 16 15 14 13 12 11 | 21 22 23 24 25 26 27 28
Lower: 48 47 46 45 44 43 42 41 | 31 32 33 34 35 36 37 38
```

Verify the original radiograph's side marker before annotation. FDI is an
anatomical identity, not a horizontal slot. Impacted, tilted or displaced teeth
keep their identities. This project supports the 32 permanent FDI classes;
deciduous teeth, supernumerary teeth and implants need an explicitly agreed
annotation policy and potentially a different schema.

Reference: [ADA tooth designation table, including ISO designations](https://www.ada.org/-/media/project/ada-organization/ada/ada-org/files/publications/cdt/universal_tooth_designation_system_valueset_2.pdf).

## 1. Collect clean radiographs and expert labels

Use the original X-rays without printed tooth labels, colored masks, boxes or
reference-chart graphics. The images supplied in this conversation illustrate
the numbering; annotated screenshots should not become training inputs.

Include both present and confirmed-missing examples for the same FDI classes,
especially your target cases 16, 24 and 33. Include different combinations of
missing teeth, spacing, restorations, unerupted/impacted teeth, scanners and image
quality. Keep evaluation cases representative of the images you will upload.

A dental reviewer should identify each tooth, draw a tight axis-aligned box
around the visible whole tooth (crown and visible roots), and check neighboring
identities. Do not draw boxes over empty gaps. Agree how to annotate retained
roots, severe overlap and partial visibility before annotating the whole dataset.
Keep uncertain cases explicitly uncertain; do not guess identities from spacing.

## 2. Use this project's exact class mapping

Import `configs/annotation_classes.txt` into a YOLO-compatible annotation tool,
or consult `configs/fdi_class_mapping.csv`. The full mapping also appears in
`configs/data.yaml`. The numeric YOLO class ID is different from the FDI code:

| FDI | YOLO class ID | Example policy |
| --- | ---: | --- |
| 15 | 4 | Box only if tooth 15 is visible |
| 16 | 5 | If missing, no box and no row with class 5 |
| 17 | 6 | Remains class 6 even when 16 is missing |
| 18 | 7 | Remains class 7 even when 16 is missing |
| 24 | 11 | If missing, no box and no row with class 11 |
| 25 | 12 | Remains class 12 even when 24 is missing |
| 33 | 18 | If missing, no box and no row with class 18 |
| 34 | 19 | Remains class 19 even when 33 is missing |

Do not add a detector class called "missing tooth": absence has no tooth box.
The model learns object identities and background from correctly labeled images.

Export one label file per image. Each actual tooth is one normalized row:

```text
class_id center_x center_y width height
```

Divide x/width by image width and y/height by image height. For an image with
16, 24 and 33 missing, omit class IDs 5, 11 and 18 while keeping all other correct
IDs. Do not replace gaps with dummy boxes. The row order in a label file does
not define tooth identity. [Official YOLO detection format](https://docs.ultralytics.com/datasets/detect/).

## 3. Separate absence ground truth from a missed detection

A missing YOLO row is not enough to distinguish a genuinely missing tooth from
an incomplete annotation. Add expert-reviewed presence metadata when you want
to audit this distinction:

```text
dataset/presence/train/patient_001.json
dataset/presence/val/patient_002.json
dataset/presence/test/patient_003.json
```

Each JSON uses this schema and must contain all 32 FDI keys:

```json
{
  "schema": "fdi_presence_annotation_v1",
  "status": {
    "11": "present",
    "12": "present",
    "16": "missing",
    "24": "missing",
    "33": "missing"
  }
}
```

The snippet is abbreviated: add the other 27 keys before use. Allowed values
are `present`, `missing`, and `uncertain`. Use `uncertain` for an unresolved or
out-of-field tooth rather than inferring absence. Metadata uses the same image
stem as the X-ray. The validator checks that missing teeth have no boxes and
present teeth have boxes. These records support annotation checks; they do not
automatically train a dedicated absence classifier or make missing diagnoses.
The existing trainer rejects completely empty label files for explicit review.

In the app, **Confirmed missing teeth (reviewed)** records a human decision,
removes conflicting displayed boxes, and never renumbers neighbors. An
automatic nondetection remains **Not detected — review**. If a tooth is present
but missed, add/correct its box instead of marking it missing. Bulk review applies
missing selections only to the selected image. Undoing a selection restores the
original prediction without rerunning inference.

## 4. Split by patient before training

```text
dataset/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
  presence/train/ presence/val/ presence/test/   # optional reviewed metadata
```

An initial split can be 80% train, 10% validation, 10% independent test. All scans
and variants from a patient stay in one split. Keep a patient-ID split manifest;
the current validator checks exact duplicate images, not patient identity.
Never copy or augment the same image across splits. Use validation for threshold
and fusion choices; keep test patients untouched until the final comparison.

There is no image count that guarantees 90% accuracy. Current local coverage is
only 24 training images (635 annotated teeth), 5 validation images, and no test
images. FDI 33 is annotated in every training image; this dataset does not provide
a labeled no-33 training example. Audit labels rather than assuming unannotated
teeth are absent. Build a larger pilot, inspect failures, then add reviewed cases
for the weak classes and missing-tooth patterns. Crops from one patient do not
become independent patients.

## 5. Check labels, then train detection first

Run from the project folder:

```powershell
python -m dental_fdi.data.audit_dataset
python -m dental_fdi.training.train --check-only
python -m dental_fdi.training.train --task detect --epochs 100 --imgsz 1024 --batch 4 --device 0 --name fdi_missing_teeth
```

Use `--device cpu` on this machine's current CPU-only PyTorch installation;
YOLO26x training will be slow. The provided notebook can run detector fine-tuning
in a GPU runtime. Build the current local archive with
`python -m tools.build_training_bundle`; it does not upload anything.

Review `reports/evaluation/dataset_audit.json` and several rendered labels before
training. Automatic checks catch duplicate FDI classes, invalid/out-of-image
boxes, duplicate images across train/val/test, and contradictions in optional
presence metadata. They cannot verify dental anatomy or incomplete labels.

Training keeps FDI side orientation: horizontal and vertical flips are disabled.
Use original-image training inputs consistent with inference preprocessing.
The detector trainer writes a new run and reports the actual best checkpoint;
do not assume the folder name if Ultralytics adds a run suffix.

## 6. Rebuild all three models using the new detector

After reviewing detector validation, use its actual best checkpoint:

```powershell
python -m dental_fdi.training.train_ensemble --detector runs/detect/fdi_missing_teeth/weights/best.pt --epochs 100 --batch 32
python -m dental_fdi.training.evaluate_hybrid --hybrid runs/ensemble/yolo26_effnet_resnet50/model.fdi.zip --output reports/evaluation/three_model_comparison.json
```

This trains both dental classifier heads using frozen ImageNet encoders, then
packages YOLO26, EfficientNet-B0 and ResNet50 into one bundle. It does not jointly
fine-tune all networks. Both classifiers contribute at inference, but they cannot
recover tooth boxes the detector never proposed. Restart the app after replacing
the bundle. See [fusion details](THREE_MODEL.md).

## 7. Test numbering and missing-tooth errors separately

For independent testing, add `test: images/test` to `configs/data.yaml`, supply
test labels and optional presence metadata, then run:

```powershell
python -m dental_fdi.training.evaluate_hybrid --hybrid runs/ensemble/yolo26_effnet_resnet50/model.fdi.zip --split test --output reports/evaluation/three_model_test.json
```

The evaluator requires the correct FDI and IoU >= 0.5 for a correct detection.
It reports precision, recall and F1 at the chosen score threshold, not clinical
accuracy. Audit separately: wrong FDI on a real tooth, missed present tooth,
false box in a confirmed-missing location, and false absence conclusions.
Use reviewed presence records for that audit; the current evaluator measures
boxes/FDI rather than a trained absence diagnosis.

Include test images with 16 missing but 17/18 present, 24 missing but 23/25
present, 33 missing but 32/34 present, multiple gaps, complete dentitions and
impacted teeth. Inspect per-class performance, not just an overall average.
Keep the simplest pipeline that meets your held-out detection requirements;
the existing three-model bundle has not improved development F1 over YOLO alone.
