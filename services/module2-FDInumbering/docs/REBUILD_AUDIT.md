# Rebuild audit

## Scope and changes

- Audited Python imports, app/model entry points, dataset configuration, training
  tools, tests, UI assets, archived scripts, model artifacts and report locations.
- Replaced the flat source layout with `dental_fdi/{analysis,inference,imaging,
  reporting,ui,training,data}` and centralized project paths in `paths.py`.
- Kept `app.py` as the Streamlit entry point. Rebuilt imports and asset paths,
  including the custom molar-axis component. CLI tools use `python -m ...`.
- Moved configs, docs, notebooks, evaluation reports, logs, generic checkpoints
  and example images to named directories. Existing `dataset/`, `runs/` and
  model weights are retained. Old training archives are under `artifacts/legacy`.
- Added package metadata and a dependency entry point. Generated caches, local
  datasets and trained artifacts are ignored for future Git additions.
- Rebuilt notebook generation and training archive packaging for the new layout.
- Preserved guessed-label and dataset-replacement helpers as inactive text under
  `tools/legacy`. They are excluded from the active training package and bundle.
- Added one versioned three-model artifact and loader, strict positive fusion
  weights, all-model execution, full contribution traces, and CSV audit columns.
- Single and bulk UI predictions use the same inference and overlay paths.

## Verification

- Existing analysis, image-axis, hybrid, detection and export tests were migrated.
- New fusion tests check that each network changes the result, both classifiers
  run for every crop, a component cannot be assigned zero weight, and empty
  detector output never invents teeth.
- Dataset validation and detector `--check-only` pass after config relocation.
- Streamlit loads through the root entry point with the three-model default.
- Full-image evaluation uses the real bundle against its bundled detector.
- The optional browser smoke test uploads two real local validation images plus
  one invalid file, then checks downloaded ZIP CRCs, overlays, failure reporting
  and the contribution fields from all three models.

## Limits

This is an execution and organization audit, not a clinical validation of the
angulation or periodontal algorithms. Only 24 training and 5 development
validation images are present. Existing annotations have not been independently
re-adjudicated in this rebuild. Fusion did not improve development F1. No
independent test set, patient identity manifest or calibrated confidence model
is present. These limitations remain visible in documentation and app notices.
