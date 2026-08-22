#!/usr/bin/env python3
"""
Train full-arch tooth instance segmenter (AKUDENTAL-style colored masks).

Requires COCO annotations with:
  - segmentation: polygon or RLE mask per tooth
  - attributes.fdi_number: 11-48

Place annotated OPGs in training/data/raw/ and update coco_annotations.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.tooth_instance_model import build_tooth_instance_unet

FDI_PERMANENT = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
]
FDI_TO_CLASS = {fdi: idx + 1 for idx, fdi in enumerate(FDI_PERMANENT)}


def polygon_to_mask(polygon: list[float], height: int, width: int) -> np.ndarray:
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def build_instance_dataset(raw_dir: Path, ann_file: Path, image_size: int = 512):
    with open(ann_file, encoding="utf-8") as f:
        coco = json.load(f)

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
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        h, w = image.shape[:2]
        label_map = np.zeros((h, w), dtype=np.uint8)

        for ann in anns:
            fdi = ann.get("attributes", {}).get("fdi_number")
            if fdi not in FDI_TO_CLASS:
                continue
            class_id = FDI_TO_CLASS[fdi]
            seg = ann.get("segmentation")
            if not seg:
                x, y, bw, bh = ann["bbox"]
                label_map[int(y) : int(y + bh), int(x) : int(x + bw)] = class_id
                continue
            if isinstance(seg, list) and seg and isinstance(seg[0], list):
                for poly in seg:
                    mask = polygon_to_mask(poly, h, w)
                    label_map[mask > 0] = class_id
            elif isinstance(seg, list):
                mask = polygon_to_mask(seg, h, w)
                label_map[mask > 0] = class_id

        if label_map.max() == 0:
            continue

        image = cv2.resize(image, (image_size, image_size))
        label_map = cv2.resize(label_map, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        xs.append(image[..., None])
        ys.append(label_map)

    if not xs:
        return np.empty((0, image_size, image_size, 1)), np.empty((0, image_size, image_size))

    x_arr = np.asarray(xs, dtype=np.float32) / 255.0
    y_arr = np.asarray(ys, dtype=np.int32)
    return x_arr, y_arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    ann_file = Path(cfg["data"]["annotations_dir"]) / "coco_annotations.json"
    image_size = cfg["data"].get("image_size_instance", 512)

    x, y = build_instance_dataset(raw_dir, ann_file, image_size=image_size)
    if len(x) == 0:
        raise SystemExit(
            "No instance segmentation samples found.\n"
            "Annotate real OPGs in COCO format with segmentation polygons and attributes.fdi_number.\n"
            "See docs/ARCHITECTURE.md for AKUDENTAL-style annotation guide."
        )

    num_classes = len(FDI_PERMANENT) + 1  # background + 32 teeth
    model = build_tooth_instance_unet(input_shape=(image_size, image_size, 1), num_classes=num_classes)

    epochs = cfg.get("tooth_instance", {}).get("epochs", 50)
    if os.getenv("MODULE2_QUICK_TRAIN") == "1":
        epochs = min(epochs, 5)

    model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=min(4, len(x)),
        validation_split=0.2 if len(x) > 5 else 0.0,
        verbose=1,
    )

    out_path = Path("training/checkpoints/tooth_instance_segmenter.keras")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved tooth instance segmenter to {out_path}")


if __name__ == "__main__":
    main()
