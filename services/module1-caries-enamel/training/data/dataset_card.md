# Module 1 Dataset Card (Caries + Enamel Erosion)

## Expected inputs
- Panoramic X-ray images in training/data/raw/
- COCO annotations in training/data/annotations/coco_annotations.json
- Enamel masks (.npy, single-channel) in training/data/processed/enamel_masks/

## COCO attributes expected per annotation
- tooth_fdi: integer (e.g., 16, 11, 26, 36, 31, 46)
- caries_grade: one of [none, incipient, moderate, severe]
- enamel_grade: one of [none, mild, moderate, severe]

## Notes
- Split train/val/test by patient to avoid leakage.
- Keep class distribution balanced across grades.
