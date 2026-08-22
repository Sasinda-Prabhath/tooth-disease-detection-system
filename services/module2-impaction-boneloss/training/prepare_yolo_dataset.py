#!/usr/bin/env python3
"""Convert project COCO annotations into a reproducible YOLOv8 dataset split."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

FDI_TO_CLASS = {18: 0, 28: 1, 38: 2, 48: 3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("training/data/raw"))
    parser.add_argument("--coco", type=Path, default=Path("training/data/annotations/coco_annotations.json"))
    parser.add_argument("--output", type=Path, default=Path("training/data/yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.coco.exists():
        raise SystemExit(f"COCO annotation file not found: {args.coco}")
    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in coco.get("images", [])}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        fdi = annotation.get("attributes", {}).get("fdi_number")
        if fdi in FDI_TO_CLASS:
            annotations[annotation["image_id"]].append(annotation)

    usable = [image_id for image_id in annotations if (args.images / images[image_id]["file_name"]).exists()]
    if not usable:
        raise SystemExit("No labelled third-molar OPG images found. Add images and FDI 18/28/38/48 COCO boxes first.")
    random.Random(args.seed).shuffle(usable)
    split_at = max(1, round(len(usable) * (1 - args.val_ratio)))
    partitions = {"train": usable[:split_at], "val": usable[split_at:] or usable[-1:]}

    for split, image_ids in partitions.items():
        image_dir, label_dir = args.output / "images" / split, args.output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            image = images[image_id]
            source = args.images / image["file_name"]
            shutil.copy2(source, image_dir / source.name)
            labels = []
            for annotation in annotations[image_id]:
                fdi = annotation["attributes"]["fdi_number"]
                x, y, width, height = annotation["bbox"]
                labels.append(f"{FDI_TO_CLASS[fdi]} {(x + width / 2) / image['width']:.6f} {(y + height / 2) / image['height']:.6f} {width / image['width']:.6f} {height / image['height']:.6f}")
            (label_dir / f"{source.stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    dataset = {"path": str(args.output.resolve()), "train": "images/train", "val": "images/val", "names": {0: "18", 1: "28", 2: "38", 3: "48"}}
    (args.output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    print(f"Prepared {len(partitions['train'])} training and {len(partitions['val'])} validation OPGs at {args.output}")


if __name__ == "__main__":
    main()
