#!/usr/bin/env python3
"""Generate synthetic panoramic-style samples for Module 2 training bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ANGULATION_LABELS = ["Mesioangular", "Vertical", "Horizontal", "Distoangular"]
THIRD_MOLAR_FDI = {18, 28, 38, 48}

# Approximate panoramic layout (x, y, w, h as fractions of image size) per FDI
FDI_LAYOUT: dict[int, tuple[float, float, float, float]] = {}
_upper = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
for i, fdi in enumerate(_upper):
    x = 0.02 + i * 0.058 if fdi < 20 else 0.52 + (i - 8) * 0.058
    FDI_LAYOUT[fdi] = (x, 0.12, 0.05, 0.32)
_lower = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
for i, fdi in enumerate(_lower):
    x = 0.02 + i * 0.058 if fdi > 40 else 0.52 + (i - 8) * 0.058
    FDI_LAYOUT[fdi] = (x, 0.56, 0.05, 0.32)


def make_synthetic_panoramic(width: int = 1024, height: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    for _ in range(12):
        cx = int(rng.integers(width * 0.05, width * 0.95))
        cy = int(rng.integers(height * 0.2, height * 0.8))
        axes = (int(rng.integers(20, 45)), int(rng.integers(30, 70)))
        color = int(rng.integers(120, 220))
        cv2.ellipse(image, (cx, cy), axes, int(rng.integers(-30, 30)), 0, 360, (color, color, color), -1)
    return image


def bbox_to_polygon(x: float, y: float, w: float, h: float) -> list[float]:
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("training/data"))
    parser.add_argument("--num-images", type=int, default=40)
    args = parser.parse_args()

    raw_dir = args.output / "raw"
    ann_dir = args.output / "annotations"
    masks_dir = args.output / "processed" / "segmentation_masks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    images_meta = []
    annotations = []
    ann_id = 1

    for i in range(args.num_images):
        img = make_synthetic_panoramic(seed=i)
        fname = f"synthetic_{i:03d}.png"
        cv2.imwrite(str(raw_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        image_id = i + 1
        h, w = img.shape[:2]
        images_meta.append({"id": image_id, "file_name": fname, "width": w, "height": h})

        for fdi, (fx, fy, fw, fh) in FDI_LAYOUT.items():
            x = int(w * fx)
            y = int(h * fy)
            bw, bh = int(w * fw), int(h * fh)
            attrs: dict = {"fdi_number": fdi, "impacted": fdi in THIRD_MOLAR_FDI}
            if fdi in THIRD_MOLAR_FDI:
                attrs["angulation"] = ANGULATION_LABELS[i % len(ANGULATION_LABELS)]

            poly = bbox_to_polygon(float(x), float(y), float(bw), float(bh))
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, bw, bh],
                    "segmentation": [poly],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "attributes": attrs,
                }
            )
            ann_id += 1

            if fdi in THIRD_MOLAR_FDI:
                cej_mask = np.zeros((bh, bw), dtype=np.uint8)
                crest_mask = np.zeros((bh, bw), dtype=np.uint8)
                cej_y = int(bh * 0.6)
                crest_y = int(bh * 0.35)
                cv2.line(cej_mask, (5, cej_y), (bw - 5, cej_y), 1, 2)
                cv2.line(crest_mask, (5, crest_y), (bw - 5, crest_y), 1, 2)
                mask = np.stack([cej_mask, crest_mask], axis=-1).astype(np.float32)
                np.save(masks_dir / f"synthetic_{i:03d}_fdi{fdi}.npy", mask)

    coco = {
        "images": images_meta,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "tooth"}],
    }
    with open(ann_dir / "coco_annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"Generated {args.num_images} synthetic images in {raw_dir}")
    print(f"COCO annotations (32 teeth / image + segmentation polygons): {ann_dir / 'coco_annotations.json'}")


if __name__ == "__main__":
    main()
