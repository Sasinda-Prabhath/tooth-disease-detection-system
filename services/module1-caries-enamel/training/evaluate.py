#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from dataset import build_caries_dataset, build_enamel_segmentation_dataset
from dental_common.metrics import dice_coefficient, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    ann_file = Path(cfg["data"]["annotations_dir"]) / "coco_annotations.json"

    x, y = build_caries_dataset(raw_dir, ann_file)
    if len(x):
        labels = np.argmax(y, axis=1)
        preds = np.argmax(y, axis=1)
        print(f"Caries macro-F1 (sanity identity): {macro_f1(labels, preds):.3f}")

    masks_dir = Path(cfg["data"]["processed_dir"]) / "enamel_masks"
    sx, sy = build_enamel_segmentation_dataset(raw_dir, masks_dir)
    if len(sx):
        dice = dice_coefficient(sy, sy)
        print(f"Enamel Dice (sanity identity): {dice:.3f}")


if __name__ == "__main__":
    main()
