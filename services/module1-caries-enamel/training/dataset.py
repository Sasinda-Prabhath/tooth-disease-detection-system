from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CARIES_LABELS = ["none", "incipient", "moderate", "severe"]
FDI_LIST = [16, 11, 26, 36, 31, 46]


def load_coco_annotations(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_detector_samples(raw_dir: Path, annotations_file: Path, image_size: int = 640):
    coco = load_coco_annotations(annotations_file)
    images = {img["id"]: img for img in coco.get("images", [])}

    xs, boxes, classes = [], [], []
    for ann in coco.get("annotations", []):
        meta = images.get(ann.get("image_id"))
        if not meta:
            continue

        img_path = raw_dir / meta["file_name"]
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size))

        x, y, w, h = ann["bbox"]
        scale_x = image_size / max(meta.get("width", image_size), 1)
        scale_y = image_size / max(meta.get("height", image_size), 1)

        box = np.array(
            [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y],
            dtype=np.float32,
        )

        fdi = ann.get("attributes", {}).get("tooth_fdi", 16)
        cls = FDI_LIST.index(fdi) if fdi in FDI_LIST else 0

        xs.append(image)
        boxes.append(box)
        classes.append(cls)

    if not xs:
        return np.empty((0, image_size, image_size, 3)), np.empty((0, 4)), np.empty((0,), dtype=np.int32)

    x_arr = np.asarray(xs, dtype=np.float32) / 255.0
    box_arr = np.asarray(boxes, dtype=np.float32)
    cls_arr = np.asarray(classes, dtype=np.int32)
    return x_arr, box_arr, cls_arr


def build_caries_dataset(raw_dir: Path, annotations_file: Path, image_size: int = 224):
    coco = load_coco_annotations(annotations_file)
    images = {img["id"]: img for img in coco.get("images", [])}

    xs, ys = [], []
    for ann in coco.get("annotations", []):
        meta = images.get(ann.get("image_id"))
        if not meta:
            continue

        img_path = raw_dir / meta["file_name"]
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x, y, w, h = ann["bbox"]
        crop = image[int(y) : int(y + h), int(x) : int(x + w)]
        if crop.size == 0:
            continue

        label = ann.get("attributes", {}).get("caries_grade", "none")
        if label not in CARIES_LABELS:
            continue

        crop = cv2.resize(crop, (image_size, image_size))
        xs.append(crop)
        ys.append(CARIES_LABELS.index(label))

    if not xs:
        return np.empty((0, image_size, image_size, 3)), np.empty((0, len(CARIES_LABELS)))

    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.zeros((len(ys), len(CARIES_LABELS)), dtype=np.float32)
    for i, idx in enumerate(ys):
        y_arr[i, idx] = 1.0
    return x_arr, y_arr


def build_enamel_segmentation_dataset(raw_dir: Path, masks_dir: Path, image_size: int = 512):
    xs, ys = [], []
    for mask_path in sorted(masks_dir.glob("*.npy")):
        stem = mask_path.stem
        img_candidates = list(raw_dir.glob(f"{stem}.*"))
        if not img_candidates:
            continue

        image = cv2.imread(str(img_candidates[0]), cv2.IMREAD_GRAYSCALE)
        mask = np.load(mask_path)

        if image is None or mask.ndim != 2:
            continue

        image = cv2.resize(image, (image_size, image_size))
        mask = cv2.resize(mask.astype(np.float32), (image_size, image_size))

        xs.append(image[..., None])
        ys.append(mask[..., None])

    if not xs:
        return np.empty((0, image_size, image_size, 1)), np.empty((0, image_size, image_size, 1))

    x_arr = np.asarray(xs, dtype=np.float32) / 255.0
    y_arr = np.asarray(ys, dtype=np.float32)
    return x_arr, y_arr
