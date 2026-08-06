from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ANGULATION_LABELS = ["Mesioangular", "Vertical", "Horizontal", "Distoangular"]
THIRD_MOLAR_FDI = {18, 28, 38, 48}


def load_coco_annotations(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_images(raw_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in exts)


def build_angulation_dataset(
    raw_dir: Path,
    annotations_file: Path,
    image_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """Build classification dataset from COCO crops for third molars."""
    coco = load_coco_annotations(annotations_file)
    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    xs, ys = [], []
    for image_id, anns in anns_by_image.items():
        meta = images.get(image_id)
        if not meta:
            continue
        img_path = raw_dir / meta["file_name"]
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for ann in anns:
            fdi = ann.get("attributes", {}).get("fdi_number")
            if fdi not in THIRD_MOLAR_FDI:
                continue
            label = ann.get("attributes", {}).get("angulation")
            if label not in ANGULATION_LABELS:
                continue
            x, y, w, h = ann["bbox"]
            crop = image[int(y) : int(y + h), int(x) : int(x + w)]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (image_size, image_size))
            xs.append(crop)
            ys.append(ANGULATION_LABELS.index(label))

    if not xs:
        return np.empty((0, image_size, image_size, 3)), np.empty((0, len(ANGULATION_LABELS)))

    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.zeros((len(ys), len(ANGULATION_LABELS)), dtype=np.float32)
    for i, label_idx in enumerate(ys):
        y_arr[i, label_idx] = 1.0
    return x_arr, y_arr


def build_segmentation_dataset(
    raw_dir: Path,
    masks_dir: Path,
    image_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for mask_path in sorted(masks_dir.glob("*.npy")):
        stem = mask_path.stem.split("_fdi")[0]
        img_candidates = list(raw_dir.glob(f"{stem}.*"))
        if not img_candidates:
            continue
        image = cv2.imread(str(img_candidates[0]), cv2.IMREAD_GRAYSCALE)
        mask = np.load(mask_path)
        if image is None or mask.ndim != 3:
            continue
        image = cv2.resize(image, (image_size, image_size))
        mask = cv2.resize(mask, (image_size, image_size))
        xs.append(image[..., None])
        ys.append(mask)

    if not xs:
        return np.empty((0, image_size, image_size, 1)), np.empty((0, image_size, image_size, 2))

    x_arr = np.asarray(xs, dtype=np.float32) / 255.0
    y_arr = np.asarray(ys, dtype=np.float32)
    return x_arr, y_arr
