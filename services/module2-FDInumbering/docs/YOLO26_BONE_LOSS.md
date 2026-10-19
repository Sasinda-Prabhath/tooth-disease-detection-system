# YOLO26 bone-level workflow

This module implements a six-keypoint YOLO26-pose adaptation of Lee et al.,
https://pmc.ncbi.nlm.nih.gov/articles/PMC9026777/ . The paper used three U-Net
segmentation networks on periapical radiographs. This implementation does NOT
reproduce those networks or inherit their reported accuracy. Panoramic images
require their own expert protocol and validation.

## Implemented pipeline

Reviewed FDI tooth boxes -> clean padded crops -> canonical mesial/distal orientation
-> YOLO26 six-landmark model -> axial ratios -> human review -> PNG/CSV/JSON.
The app also permits manual landmark placement without a trained model.

Landmark order is fixed:
1. mesial CEJ
2. mesial bone crest
3. mesial root apex/reference
4. distal CEJ
5. distal bone crest
6. distal root apex/reference

Crop preparation mirrors quadrants 1 and 4 so mesial is on image left in training
crops. Predictions are mapped back to the working X-ray. Do not mirror these
canonical crops a second time. Check R/L markers on the whole X-ray first.

## Measurement protocol

For each side, let u be the CEJ-to-apex vector divided by its length.
Depth = dot(crest - CEJ, u).
Ratio = 100 * depth / length(apex - CEJ).
The tooth summary is max(mesial ratio, distal ratio).
The crest's lateral offset and projected point are also recorded internally.
This uses a CEJ-to-apex axis per surface; it is a landmark adaptation, not the
paper's segmentation-derived whole-tooth axis. A clinician must approve the
root-reference protocol, especially for multirooted or tilted teeth.

The cyan overlay is the projected measurement; a thin connector shows the actual
crest landmark's offset. Ratios do not establish periodontitis or clinical stage.
No millimetres are inferred from screenshots or uncalibrated PNGs. In particular,
the paper's <1.5 mm stage-0 rule cannot be applied without valid calibration.
No arbitrary physiological offset is subtracted. Missing landmarks are unavailable.
The current six-point interface requires both sides; partially assessable teeth
need a future side-specific protocol before their percentages can be reported.

## 1. Collect and annotate clean data

From this module directory:

```powershell
py -3.11 -m dental_fdi.data.prepare_bone_crops
```

Crops are prepared already if periodontal_dataset/images contains files. This
command refuses to overwrite existing crops/annotations. The crop_manifest.json
links every crop to its source radiograph. FDI detection labels are NOT bone labels.
Do not use the same image both with and without painted overlays as separate data.

A dentist/periodontist must review source tooth numbers and boxes, then label six
keypoints in CVAT or another tool that exports Ultralytics pose labels. The app's
reviewed bone training-example ZIP is another way to obtain labels and clean crops.
For obscured or ambiguous landmarks do not invent positions. Visibility 0 means
unavailable, 1 means labeled but occluded, 2 means visibly annotated. Evaluation
uses visibly annotated points and reports unavailable measurements.

One label per crop, matching its image stem:

class cx cy width height mCEJx mCEJy vis mCrestx mCresty vis mApexx mApexy vis dCEJx dCEJy vis dCrestx dCresty vis dApexx dApexy vis

Coordinates and boxes are normalized to crop width/height. The class is 0
(tooth_region), not the FDI class. A full-crop box is `0 0.5 0.5 1 1` followed
by 18 keypoint values. All available landmarks must be inside that region box.
Keep enough context to see the crest and the relevant apex; fix the source box
if the crop omits them. Agree with the clinician how crowns/restorations, implants,
third molars, missing teeth and multirooted teeth are handled. Do not silently
replace an obscured CEJ with a restoration margin unless that policy is documented.

## 2. Split by PATIENT, not by crop

Use approximately 70/15/15 train/validation/test as a starting split, adjusted to
cohort size and agreed evaluation design. All teeth, radiographs and repeat visits
from a patient must stay in one split. Existing crop split assignments only follow
the source dataset: they do NOT prove patient independence. Reassign before training.

Directory structure:

periodontal_dataset/
  images/train, images/val, images/test
  labels/train, labels/val, labels/test
  patients.csv
  crop_manifest.json

Create patients.csv with one row per image, using real pseudonymous patient mapping:

```csv
image,patient_id,split
some_crop_FDI11.png,P001,train
another_crop_FDI26.png,P002,val
held_out_crop_FDI36.png,P003,test
```

Do not assign a new patient ID to each tooth. Never fabricate patient identity
from filenames unless the dataset provider confirms that mapping.

## 3. Check and train YOLO26

```powershell
py -3.11 -m dental_fdi.training.train_periodontal --check-only
py -3.11 -m dental_fdi.training.train_periodontal --weights yolo26s-pose.pt --epochs 100 --batch 4 --imgsz 512 --device 0 --name periodontal_landmarks
```

Use `--device cpu` when no supported GPU is available; training may be slow.
The first training run may download generic YOLO26 pose weights. They are an
initialization, not a pretrained dental bone-loss model. The trainer adapts the
head to one class and six keypoints from the dataset config. Test labels are
required for readiness checks, but the training loop uses only train/validation.
Output is a candidate under runs/pose. Inspect the actual printed path because
Ultralytics may increment an existing run name. Do not overwrite earlier runs.
The bone_schema.json and bone_training.json accompany best.pt.

## 4. Evaluate before enabling proposals

```powershell
py -3.11 -m dental_fdi.training.evaluate_periodontal --weights runs/pose/periodontal_landmarks/weights/best.pt --dataset periodontal_dataset --output reports/evaluation/periodontal_yolo26.json
```

The report contains landmark error normalized to crop diagonal, mesial/distal
ratio MAE in percentage points, and measured/unavailable counts. Missing predictions
are NOT zero-error measurements. Report coverage and errors together. Review
failure cases, examiner agreement, clinically important strata and image modalities.
Choose acceptance criteria with the supervising clinician BEFORE test evaluation;
never tune the model on the held-out test set. This evaluator does not establish
clinical diagnostic validity or reproduce the paper's reported metrics.

## 5. Open the trained candidate in the app

In the same PowerShell window used to start Streamlit:

```powershell
$env:FDI_BONE_MODEL_PATH = (Resolve-Path 'runs/pose/periodontal_landmarks/weights/best.pt').Path
py -3.11 -m streamlit run app.py
```

Use Bone levels. Proposed landmarks are unreviewed. Correct the CEJ, crest and
apex, then confirm the six landmarks to include reviewed results in the default
overlay. User edits are kept separately from model proposals. Downloads retain
source, review status, measurement definition and unavailable values. This does
not restore the removed automatic wisdom-tooth angle section.

## Current limitations

No trained dental bone-landmark weights or expert landmark labels were available
when this workflow was implemented. Prepared crops alone cannot train the model.
The checker must fail until labels, patient mapping and all three splits exist.
No claim of model accuracy, clinical staging, patient diagnosis or millimetre
measurement can be made from the implementation tests.

Sources:
- Lee et al.: https://pmc.ncbi.nlm.nih.gov/articles/PMC9026777/
- YOLO26: https://docs.ultralytics.com/models/yolo26/
- Pose labels: https://docs.ultralytics.com/datasets/pose/
