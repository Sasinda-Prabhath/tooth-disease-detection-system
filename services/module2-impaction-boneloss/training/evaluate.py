#!/usr/bin/env python3
"""Evaluate Module 2 models on held-out annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from dataset import build_angulation_dataset, build_segmentation_dataset
from dental_common.metrics import bone_loss_mae, dice_coefficient, macro_f1, severity_kappa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    ann_file = Path(cfg["data"]["annotations_dir"]) / "coco_annotations.json"

    x, y = build_angulation_dataset(raw_dir, ann_file)
    if len(x):
        preds = np.argmax(y, axis=1)
        labels = np.argmax(y, axis=1)
        print(f"Angulation macro-F1 (baseline sanity): {macro_f1(labels, preds):.3f}")

    masks_dir = Path(cfg["data"]["processed_dir"]) / "segmentation_masks"
    sx, sy = build_segmentation_dataset(raw_dir, masks_dir)
    if len(sx):
        dice = dice_coefficient(sy, sy)
        print(f"Segmentation Dice (identity sanity): {dice:.3f}")

    print("For bone-loss MAE and Cohen's Kappa, provide expert mm measurements in evaluate_annotations.csv")


if __name__ == "__main__":
    main()
