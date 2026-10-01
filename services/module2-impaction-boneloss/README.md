# Module 2: third-molar OPG research

Python 3.11 / FastAPI service with React + TypeScript UI in the repository's existing `frontend/`. The service lives in `app/` to preserve gateway and startup integration; it does not duplicate the frontend under this directory.

**Implementation is a research scaffold, not a trained or clinically validated system.** No patient data, annotations, OCR exports, or dental weights are included. Real-model inference, model quality, GPU compatibility and dentist validation must be established with your artifacts. Backend tests exercise numerical contracts and orchestration with explicit test doubles; they are not clinical accuracy evidence.

## Run

From repository root on Windows:

```powershell
.\scripts\setup-module2.ps1
.\scripts\start-module2.ps1
```

Open http://localhost:3000. Base dependencies run the API and UI with `degraded` health. To install model runtimes, use `setup-module2.ps1 -ML`, provision the artifacts below, then restart. PaddleOCR uses its own Paddle runtime; the dental models use PyTorch. Install hardware-compatible PyTorch/Paddle builds when enabling CUDA.

Alternatively:

```sh
docker compose -f docker-compose.module2-only.yml up --build
# Full ML dependencies and NVIDIA runtime:
docker compose -f docker-compose.module2-only.yml -f docker-compose.module2-gpu.yml up --build
```

The base Docker image intentionally omits large ML runtimes. PostgreSQL is enabled by Compose. Local development defaults to SQLite for metadata. Optional private S3 storage uses `S3_BUCKET`, `S3_ENDPOINT_URL` and normal AWS credentials; originals always stay in local protected storage. Compose credentials are development defaults.

## Runtime artifacts

| Location under `models/` | Required content |
| --- | --- |
| `privacy/det/`, `privacy/rec/` | Official PP-OCRv6 medium exported inference directories, including `inference.yml` and model parameters |
| `fdi_yolo/best.pt` | Fine-tuned YOLO26l-seg, exactly 32 FDI class names in quadrant/digit order |
| `impaction/best.pt` | `training.train_impaction` checkpoint, EfficientNetV2-S + 12-feature geometry MLP |
| `bone_loss/` | nnU-Net trained 2D folder with `dataset.json`, `plans.json`, `fold_0/checkpoint_final.pth` |

COCO weights are useful for training initialization only; the FDI adapter rejects them. Models are loaded once on startup without weight downloads. Health becomes `ok` only when all four adapters load successfully. Runtime failures mark the affected adapter unavailable. SHA-256 artifact fingerprints are returned as model versions. Restart after replacing artifacts.

## Inference contract

The workflow is **upload → privacy preparation → FDI numbering and numbered preview → third-molar impaction/angulation and preview → alveolar bone-loss assessment and preview → final report**. All third-molar impaction checks finish before bone segmentation starts. Downstream models receive the same clean processed radiograph crops plus FDI coordinates; drawn labels do not become model input. Bone-loss assessment currently covers third-molar regions, not a full-mouth periodontal assessment.

The UI uses `POST /module2/workflow` with the same upload fields and session header as `/predict`. This streams newline-delimited JSON events: `stage` and `result` events contain a `prediction`, and `error` events contain a `detail`. Stage snapshots expose ordered `workflow` entries with `pending`, `running`, `completed`, `partial`, `blocked`, or `failed` status. The gateway forwards the stream without buffering. An HTTP 200 stream can still end with an `error` event; clients must require a terminal `result` to consider the workflow finished. The existing `/predict` endpoint still returns one final JSON response.

Previews are available through protected `fdi-preview`, `impaction-preview`, and `bone-loss-preview` case artifacts. The page automatically displays previews as they arrive; selecting a tab keeps that preview selected. The final JSON report includes all findings and stage outcomes, including blocked stages when weights or reliable FDI crops are unavailable. Privacy failures withhold processed images and assessments but still produce a status report. Neither the top-level original-image URL nor the nested upload-stage original-image URL is included in that report. Bone segmentation can proceed after an uncertain or failed impaction assessment because it depends on FDI crops, not the classifier's output. Missing calibration or unassessable sites are reported as partial assessments.

`POST /module2/predict` accepts multipart `file` and optional verified isotropic `pixel_spacing_mm`. `GET /module2/health` reports model availability. `/predict` and `/health` remain compatibility aliases. The existing gateway forwards the canonical routes and protected artifacts.

Send `X-Case-Session` with a cryptographically random 64-character hex value on upload and artifact requests. The browser generates it in memory. A case is accessible only with the same capability until expiry; there are no public static mounts, raw URLs in database records, or URL-embedded tokens. This is temporary capability isolation, **not institutional identity authentication**. Place an authenticated TLS gateway, request-body limits and rate limits in front of shared deployments. The repository's unrelated gateway auth placeholder is not production authentication.

Original rendered pixels are kept in `outputs/raw_protected/` for one hour by default. The uploaded DICOM binary and its metadata are not persisted. Uploaded filenames may themselves contain identifiers and are stored only in protected metadata. Files and case metadata expire together. Cleanup runs on startup and every minute while the service runs; a stopped service cannot perform timed physical deletion. Production should additionally enforce storage lifecycle retention. Configure `MODULE2_STORAGE_DIR`, `DATABASE_URL`, `CASE_TTL_SECONDS`, `CORS_ORIGINS`, `MODELS_DIR`, `TORCH_DEVICE` as needed. Do not place runtime storage in a publicly served or cloud-synchronised directory for real patient data; this checkout is under OneDrive.

Processing uses grayscale → OCR mask → CLAHE (2.0, 8×8) → NLM (h=10) → 0–255 normalization → RGB uint8. Masked pixels are restored to zero after enhancement. Text detections near borders are masked; interior detections block downstream processing for manual review. OCR is not proof of de-identification. If OCR is unavailable, the protected original and a status report are returned; predictions are withheld.

Models receive clean processed pixels plus coordinates. Rendered boxes, contours, labels and Grad-CAM are UI artifacts only. FDI duplicates retain the highest confidence; low-confidence, jaw-order and sequence conflicts stay `uncertain` and are not passed into the third-molar models. No absent tooth is inferred. The anatomical convention assumes patient right appears on image left; mirrored or rotated images require review.

Third-molar crops include the adjacent second molar where detected. Geometry is versioned and shared with training; polygon PCA gives an **undirected** axis, not a verified root/crown direction. Impaction and angulation confidence cutoffs (0.70) and FDI confidence (0.50) are research defaults needing validation. Angulation is never returned for non-impacted teeth.

Bone distance uses `sqrt((dx * column_spacing)^2 + (dy * row_spacing)^2)`. The response returns spacing as `[row, column]`, preserving anisotropic DICOM calibration. `ImagerPixelSpacing` and guessed tooth dimensions are not substitutes. DICOM calibration still requires acquisition review, especially with panoramic projection distortion. Missing spacing yields null mm values. Apex landmarks remain null because a CEJ/crest segmenter does not predict them. Severity always remains `Not assessed`.

**Landmark limitation:** per-surface component centroids near the target tooth are an unvalidated heuristic. Ambiguous components, absent landmarks and substantially tilted/horizontal tooth axes abstain. This does not supply a clinically validated CEJ-to-crest correspondence algorithm. A dentist-reviewed surface/landmark model and held-out error study are required before accepting measurements.

## Validation

```sh
python -m pytest tests -q
# From repository frontend/:
npm run typecheck
npm run build
```

See [the ordered annotation/training protocol](training/HYBRID_TRAINING.md) for real-data prerequisites, training commands and evaluation gates.

API references used: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/), [PP-OCRv6](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html), [nnU-Net inference source](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py). Prediction uses Ultralytics' default head/postprocessing; it does not override NMS without a validated export-specific reason.
