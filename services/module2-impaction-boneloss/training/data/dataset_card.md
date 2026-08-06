# Module 2 Dataset Card

## Source
- Primary: Panoramic dental radiographs (OPG) collected and annotated by Lakshitha R.M.S.K (IT23222618)
- Bootstrap: Synthetic samples generated via `training/generate_sample_data.py` for pipeline testing

## Annotation Format
- COCO JSON: `training/data/annotations/coco_annotations.json`
- Per third molar: FDI number, angulation class, impacted flag
- Segmentation masks: `.npy` files with CEJ (channel 0) and crest (channel 1) lines

## License / IRB
- Update with institutional IRB approval reference before production use.

## Counts
- Run `python training/generate_sample_data.py` to create 40 synthetic bootstrap images.
- Replace with real annotated OPG dataset for final evaluation.
