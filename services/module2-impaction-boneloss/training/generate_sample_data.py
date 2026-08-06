#!/usr/bin/env python3
"""Generate synthetic panoramic-style samples for Module 2 training bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ANGULATION_LABELS = ["Mesioangular", "Vertical", "Horizontal", "Distoangular"]
THIRD_MOLAR_FDI = [18, 28, 38, 48]


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

        for idx, fdi in enumerate(THIRD_MOLAR_FDI):
            if fdi in (18, 38):
                x = int(w * 0.03)
            else:
                x = int(w * 0.87)
            y = int(h * (0.15 if fdi in (18, 28) else 0.55))
            bw, bh = int(w * 0.1), int(h * 0.28)
            ang = ANGULATION_LABELS[i % len(ANGULATION_LABELS)]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "attributes": {
                        "fdi_number": fdi,
                        "angulation": ang,
                        "impacted": True,
                    },
                }
            )
            ann_id += 1

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
        "categories": [{"id": 1, "name": "third_molar"}],
    }
    with open(ann_dir / "coco_annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"Generated {args.num_images} synthetic images in {raw_dir}")
    print(f"COCO annotations: {ann_dir / 'coco_annotations.json'}")


if __name__ == "__main__":
    main()
