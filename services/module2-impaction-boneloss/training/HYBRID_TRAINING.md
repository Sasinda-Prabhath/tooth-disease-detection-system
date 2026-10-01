# Ordered research protocol

Run commands from the module directory after installing `requirements-ml.txt`. These commands require real, consented, expert-labelled data. There is no train-all shortcut or synthetic accuracy claim. Patient IDs must be stable pseudonyms, including repeated scans across visits. Freeze and audit splits before training any model.

1. **Patient split.** Create `data/splits/patients.csv` with `patient_id,image` columns. Images use relative paths. Deduplicate repeated pixel data and patient aliases before splitting. Run:
   ```sh
   python -m training.prepare split data/splits/patients.csv data/splits/manifest.json
   ```
   Seed 20261001 creates approximately 70/15/15 train/validation/test patient partitions. Existing split files are never overwritten. Small sets are unsuitable for clinical conclusions.

2. **CVAT FDI annotation.** Produce privacy-reviewed, preprocessed images with the same preprocessing as inference. In CVAT annotate each visible permanent tooth using polygon label `tooth`; attribute `fdi_number` takes 11–18, 21–28, 31–38, 41–48. Add `impacted` (unknown/true/false) and `angulation` (unknown/Mesioangular/Vertical/Horizontal/Distoangular). Do not horizontally flip FDI-labelled data without an explicit relabelling scheme. Export CVAT for images XML and convert:
   ```sh
   python -m training.prepare cvat annotations.xml data/splits/manifest.json data/deidentified data/fdi_segmentation/yolo
   ```
   Converter checks dimensions, class values, polygon bounds, duplicate FDI labels, full manifest coverage and duplicate image pixels. An incomplete export is an error. Inspect outputs before training; failed conversion can leave a partial output directory, so use a fresh output path after correction.

3. **Train and validate FDI.**
   ```sh
   python -m training.train_fdi data/fdi_segmentation/yolo/dataset.yaml --device 0
   ```
   MLflow records seed and validation mAP@0.5/precision/recall. Review per-class errors and dentist numbering checks. Copy the selected run's `weights/best.pt` to `models/fdi_yolo/best.pt`. Initial COCO weights are downloaded only by this training command. Freeze thresholds on validation patients.

4. **Review processed and FDI UI.** Provision official local PP-OCRv6 exports under `models/privacy/det` and `rec`. Start API/UI. Inspect privacy masking and FDI boxes/contours before downstream training. Do not reuse failed privacy cases.

5. **Create clean third-molar crops.** Supply reviewed structured annotations indexed by image path:
   ```json
   {
     "example.png": {
       "teeth": [{"fdi":"48","confidence":1.0,"box_xyxy":[30,110,80,180],"polygon":[[30,110],[80,110],[80,180],[30,180]]}],
       "labels": {"48":{"impacted":true,"angulation":"Mesioangular"}}
     }
   }
   ```
   These coordinates illustrate the format only. Unknown labels are null; never treat unknown as a negative. CVAT attributes need expert-reviewed conversion into this structure. Then:
   ```sh
   python -m training.prepare_rois data/splits/manifest.json annotations.json data/deidentified data/impaction_angulation/crops
   ```
   Crops and 12 geometry features use the exact inference function. Evaluate the eventual classifier with predicted FDI ROIs as well as expert ROIs to measure error propagation.

6. **Train impaction + angulation.**
   ```sh
   python -m training.train_impaction data/impaction_angulation/crops/rows.json --device cuda
   ```
   EfficientNetV2-S ImageNet initialization, geometry MLP, two-class impaction and four-class angulation heads. Angulation loss excludes non-impacted/unknown cases. Validation loss selects `models/impaction/best.pt`; test rows are excluded. Select thresholds on validation data, accounting for class imbalance and abstention coverage.

7. **CEJ/crest annotation and nnU-Net.** In CVAT record `mesial_cej`, `mesial_crest`, `mesial_apex`, `distal_cej`, `distal_crest`, `distal_apex`, `crown` points for landmark validation. Missing points are null. **Additionally annotate dense CEJ and crest masks**: point annotations alone cannot train the specified segmentation task. Export single-channel PNG ROI masks matching crop dimensions and filenames: background=0, CEJ=1, crest=2. Do not invent masks by connecting sparse points without expert review.
   ```sh
   python -m training.prepare_bone data/impaction_angulation/crops/rows.json data/bone_loss/masks data/bone_loss/nnUNet_raw/Dataset502_ThirdMolar
   ```
   Set `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` to private dataset directories using environment variables. Then:
   ```sh
   nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity
   ```
   Copy generated `Dataset502_ThirdMolar/splits_final.json` to the corresponding **preprocessed** dataset directory before training. This preserves patient-level fold 0 rather than nnU-Net's automatic case split.
   ```sh
   nnUNetv2_train 502 2d 0
   ```
   Copy `dataset.json`, `plans.json`, and `fold_0/checkpoint_final.pth` from the trained 2D model folder to `models/bone_loss/`. This contract uses grayscale NaturalImage2DIO and synthetic pixel-grid spacing `[999,1,1]` for network preprocessing; original physical spacing is applied only after inference, without resizing the measurement coordinates.

8. **Calibration.** Verify DICOM PixelSpacing with the acquisition protocol. For JPEG/PNG, supply independently verified isotropic calibration or retain null mm outputs. Validate anisotropic row/column spacing and projection error; never estimate scale from tooth size.

9. **Merged reports.** Review four UI stages, per-tooth uncertainty, null measurements, provenance, Grad-CAM and JSON. Patient data and artifacts are gitignored. No whole-mouth periodontal stage or automatic severity is produced.

10. **Locked patient test.** After model/threshold selection:
    ```sh
    python -m training.train_fdi data/fdi_segmentation/yolo/dataset.yaml --evaluate models/fdi_yolo/best.pt --device 0
    python -m training.evaluate heldout_results.json metrics.json --truth-masks heldout_masks --predicted-masks predicted_masks
    ```
    Held-out results are dentist-linked records with `patient_id`, `split:"test"`, and nullable `impaction_true/pred`, `angulation_true/pred`, `bone_loss_true_mm/pred_mm`, `severity_true/pred`. Compute predictions from the frozen pipeline, retain failures/nulls, and independently audit against the frozen manifest; the metrics utility cannot prove provenance of an external file. Report impaction precision/recall/F1, angulation accuracy/macro F1/confusion matrix, CEJ and crest Dice using `training.evaluate.dice`, bone MAE in mm, and abstention coverage. Kappa is only meaningful after dentist-approved severity criteria; otherwise null. Empty masks are not automatically perfect Dice. Do not select models on test results.

11. **Dentist error review.** Record missed/duplicated FDI, OCR failures, tilted/horizontal teeth, poor CEJ visibility, calibration errors and subgroup performance. The current centroid heuristic is not validated for clinical measurements and abstains on tilted/horizontal axes. Approve a correspondence method with the dentist before trusting measured distances. No accuracy or readiness claim follows from software tests.
