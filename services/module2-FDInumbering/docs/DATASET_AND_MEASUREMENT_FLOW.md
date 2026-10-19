# Clean images, reviewed boxes, third-molar angles and bone measurements

## Which images to use

Use **clean original panoramic X-rays** as model inputs. Do not train with
numbers, colored masks, bounding boxes, green X/Y lines or red lines painted
into the images. Store boxes and landmarks separately in label files. Drawing
overlays are for checking and presentation; they are not the next model's input.

One master collection can support all tasks, provided it has the appropriate
annotations. You do not need three unrelated collections. You do need many
representative patients, not a single image, and different label schemas:

| Task | Input pixels | Required labels |
| --- | --- | --- |
| FDI numbering | Clean full panoramic image | One actual-tooth box + correct FDI for every visible supported permanent tooth |
| Third-molar axes | The same clean full panoramic images | Boxes for visible target molars + root and crown axis endpoints |
| Bone landmarks | Clean tooth-and-surrounding-bone crops from those images | Mesial and distal CEJ, bone crest and root apex, with visibility |

Keep the same patient split across every task. A patient used for training FDI
must not appear in the final held-out end-to-end test set for bone/angles.
All crops, scans and edited copies from one patient belong in the same split.
Independent public datasets can be added after checking licensing, annotation
definitions, patient duplication and scanner/image differences.

## Exact folder locations

The project root is:

```text
C:/Users/ASUSa/OneDrive/Documents/FDI Numbering system/
```

```text
dataset/                         FDI detection and numbering
  images/train/patient001.png     CLEAN full panoramic image
  images/val/patient101.png
  labels/train/patient001.txt     32-class YOLO tooth boxes, separate from pixels
  labels/val/patient101.txt
  presence/train/patient001.json  optional expert presence/missing/uncertain labels

pose_dataset/                    Third/second-molar root-to-crown axes
  images/train/patient001.png     same CLEAN panorama, same patient split
  images/val/patient101.png
  labels/train/patient001.txt     8-class boxes + TWO keypoints per molar
  labels/val/patient101.txt

periodontal_dataset/             Bone measurement landmarks
  images/train/patient001_FDI16.png  CLEAN crop INCLUDING surrounding bone
  images/val/patient101_FDI26.png
  labels/train/patient001_FDI16.txt  ONE region class + SIX keypoints
  labels/val/patient101_FDI26.txt
  crop_manifest.json             maps crops back to source panoramas and FDI
```

Test directories use the same `images/test` and `labels/test` convention. Create
them when you have real independent test images/labels, not as a replacement for
validation. Source panoramas may be copied or hardlinked between task datasets;
their label files are different and cannot be reused unchanged.

Empty train/val folders for `pose_dataset` and `periodontal_dataset` have been
created locally. They contain no invented landmarks. Their trainers will refuse
to run until valid annotations exist.

## App flow now implemented

1. Upload a clean X-ray, check the side marker and correct mirroring if required.
2. Run FDI detection. Open **Draw a tooth box on the X-ray**, select an FDI and
   drag a rectangle around a visible tooth. Press **Save drawn box for selected
   FDI**. This adds/replaces that FDI only; neighbors retain their codes.
3. Use the numeric correction editor for precise coordinates/label changes.
   Mark a truly absent tooth as confirmed missing; do not draw a gap box.
4. The canonical reviewed boxes and clean oriented image go to both third-molar
   and bone measurement stages. Boxes/lines are never burned into those inputs.
5. Optional X/Y/red drawing measures geometric line angles. Third-molar review
   separately measures the third-molar axis against the adjacent second molar.
6. Bone review lets you place six anatomical landmarks on the selected tooth
   crop, calculates pixel ratios, and requires review before exporting results.
   A trained bone model can propose points once installed. Without it, the app
   explicitly says automatic bone prediction is unavailable.
7. Download overlays for presentation, JSON/CSV for measurements, or reviewed
   training-example ZIPs containing clean pixels plus separate annotations.

Training-example exports are not automatically inserted into train/validation.
Check them with a dentist, associate the source patient ID, and place their
`images`, `labels` and optional `presence` files into the appropriate split.
The same exported image must not be placed in two splits.

## Your screenshot: X/Y/red lines versus Winter's angle

The green horizontal/vertical guides and red diagonal can define geometric
angles. The app's **Draw green X/Y guides and a red line** tool measures the
acute angle from the red line to each green line, and the angle between X and Y.
There is no requirement that the lines intersect on the image; their directions
determine the angle. Distance or "space" between lines is not an angle.

However, a vertical line in the corner of the image is not automatically an
anatomical reference. For Winter's angulation, the clinical reference is the
adjacent second molar's longitudinal axis, not arbitrary image horizontal or
vertical. The marked tooth axis also needs to follow the anatomical long axis.

For example, for lower-left 38, select 38/37 in the anatomical-axis review and
click: third root, third crown, second root, second crown. Root-to-crown direction
matters for mesial/distal signs and inversion. Upper teeth and unusual anatomy
need the clinician's agreed protocol. Angulation alone does not establish that a
tooth is impacted. [Study defining the second/third-molar axis angle](https://pmc.ncbi.nlm.nih.gov/articles/PMC11607221/).

## Step A — FDI labels and detector training

Use `configs/annotation_classes.txt` or `configs/fdi_class_mapping.csv`. FDI 16
is YOLO class 5; 24 is class 11; 33 is class 18. Missing teeth get no box. Never
shift subsequent identities to close a gap.

Each line in a detection label is:

```text
class_id center_x center_y width height
```

Coordinates are normalized to the full image. Annotate all visible supported
teeth in each training image, not just the difficult tooth.

```powershell
python -m dental_fdi.data.audit_dataset
python -m dental_fdi.training.train --task detect --check-only
python -m dental_fdi.training.train --task detect --epochs 100 --imgsz 1024 --batch 4 --device 0 --name fdi_reviewed
```

This computer currently has CPU-only PyTorch. Use `--device cpu` locally, or a
suitable GPU runtime with CUDA PyTorch. GPU `--device 0` does not work with a
CPU-only installation. The existing notebook is in `notebooks/`.

Once evaluated, a new dental detector can be packaged with the two FDI classifiers:

```powershell
python -m dental_fdi.training.train_ensemble --detector runs/detect/fdi_reviewed/weights/best.pt --epochs 100
```

The classifier combination does not itself predict bone landmarks or third-molar
root/crown points. Those are separate supervised tasks below.

## Step B — Third-molar axis labels and training

Use `configs/pose_data.yaml`: class order is **17, 18, 27, 28, 37, 38, 47, 48**.
Each visible target molar needs its actual tooth box, root-axis center and
crown-axis center. Do not derive endpoints from box corners or image vertical.
Root/crown annotation definitions must be agreed with your dentist, especially
for multiple roots, curved roots and inverted teeth.

Each row contains 11 values:

```text
class cx cy w h root_x root_y 2 crown_x crown_y 2
```

Coordinates are normalized to the full panorama. The present axis trainer
requires both landmarks visibly annotated (`2`). Exclude unassessable examples
or revise the protocol before using invisible points. Annotate all visible target
molars in an included image, including second molars; unlabeled visible targets
would be treated as background. Never label a missing wisdom tooth with a box.

```powershell
python -m dental_fdi.training.train --task pose --check-only
python -m dental_fdi.training.train --task pose --weights yolo26s-pose.pt --epochs 150 --imgsz 1024 --batch 4 --device 0 --name molar_axes
```

Install/use the actual run's `best.pt` together with `axis_schema.json`.
Default app path: `runs/pose/molar_axes/weights/best.pt`. Override a different
actual run path with `FDI_AXIS_MODEL_PATH`. Ultralytics may suffix run names;
use the actual checkpoint printed at completion.

## Step C — Bone landmarks and training

To generate clean crops from the expert-reviewed FDI dataset:

```powershell
python -m dental_fdi.data.prepare_bone_crops --source dataset --output periodontal_dataset
```

This creates clean crops with 25% context around each tooth box, preserves the
source split, and writes `crop_manifest.json`. **It creates no landmark labels.**
Quadrants 1 and 4 are mirrored in these derived training crops so the mesial
side is consistently on image left. The manifest records the transform.
Inference uses the same FDI-dependent transform and maps predicted points back
to the original panorama. The manual review screen stays in original orientation.
Do not run it over an already annotated crop folder; use a new output location
and matching config when regenerating.

With your dentist, annotate in this exact order:

1. Mesial CEJ.
2. Mesial bone crest.
3. Mesial root apex.
4. Distal CEJ.
5. Distal bone crest.
6. Distal root apex.

For single-rooted teeth, the two apex landmarks may coincide where appropriate.
Do not guess an obscured CEJ or crest from a fraction of the tooth box. A public
dataset with only tooth boxes or a mild/moderate/severe label cannot directly
train these anatomical measurements; it needs these keypoints.

The region class is **0: tooth_region**. The original FDI is retained in the
manifest and inherited from the reviewed tooth box during inference. This avoids
asking a crop-only bone model to infer the patient's quadrant again.

Each crop label has 23 values:

```text
0 0.5 0.5 1 1 mx_cej my_cej v mx_crest my_crest v mx_apex my_apex v dx_cej dy_cej v dx_crest dy_crest v dx_apex dy_apex v
```

The first five values describe the whole context region; they are not guessed
anatomical tooth boundaries. Keypoint x/y coordinates are normalized to the
**crop**, not the original panorama. Use `2` for visible annotated points. An
unlabeled/unassessable point can use `0 0 0`; never invent its position. Visibility
`1` is reserved for an occluded point whose location was independently annotated,
not a guessed location. [Official pose label format](https://docs.ultralytics.com/datasets/pose/).

You can instead mark and review the six points in the app and download **this
reviewed bone training example**. It exports the clean crop and a correctly
normalized six-point label. All six points must be reviewed and inside the crop.
Review-assisted exports still require dental annotation quality checks.

```powershell
python -m dental_fdi.training.train --task periodontal --check-only
python -m dental_fdi.training.train --task periodontal --weights yolo26s-pose.pt --epochs 150 --imgsz 512 --batch 16 --device 0 --name periodontal_landmarks
```

Default app path: `runs/pose/periodontal_landmarks/weights/best.pt`, accompanied
by `bone_schema.json`. Set `FDI_BONE_MODEL_PATH` when the actual run differs.
The app predicts points on the same context-crop geometry used in training,
then maps them back to original-image coordinates. Low-confidence points are
left unavailable for manual review. No such trained bone checkpoint is currently
installed; these commands need your actual landmark annotations first.

After installing that checkpoint, the bone-loss dashboard automatically predicts
all corrected tooth boxes. It shows measurement/review/unavailable counts,
the maximum ratio, switchable CEJ/crest/apex overlays and a per-tooth table.
Predictions are cached for the image, boxes and checkpoint version. Manual
corrections take precedence. Unreliable or geometrically invalid results remain
unavailable; they are never replaced by a box-based estimate or a zero ratio.
The PNG, CSV and JSON exports preserve unreviewed/reviewed status. Review the
six landmarks before exporting a training example.

The manual measurement currently reports CEJ-to-crest / CEJ-to-apex Euclidean
distance for each side, without a physiological offset or assumed mm scale.
Have your clinician approve the definition and any site-specific exclusions.
No automatic clinical stage, furcation grade, or proof of active bone loss is
derived. Geometry can be correct while anatomical marks are wrong.

## Check accuracy before calling the system accurate

Use independent patient-level test data, with multiple dentists adjudicating
landmarks. Measure correct-FDI box precision/recall, angle error in degrees,
bone-level error in percentage points and agreement with expert measurements.
Report unassessable/rejected cases as well as assessable cases. Include missing
teeth, impacted teeth, overlap, restorations and different scanners.

Avoid random flips that change FDI side or landmark meaning. Preserve image
aspect ratio and use original coordinates for all geometry. Physical pixel
aspect ratio and panoramic distortion need consideration; arbitrary resizing
or an uncalibrated PNG cannot establish real-world millimetres.

No dataset size or architecture guarantees complete accuracy. The current 24
FDI training panoramas do not contain expert root/crown or bone keypoint labels.
Manual drawing works now; automatic landmark predictions require the separately
annotated training datasets and independent validation.
