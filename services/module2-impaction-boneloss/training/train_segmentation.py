#!/usr/bin/env python3
"""Train bone landmark U-Net segmenter (Stage C)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_segmentation_dataset
from models.segmentation_model import build_bone_landmark_unet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    masks_dir = Path(cfg["data"]["processed_dir"]) / "segmentation_masks"
    image_size = cfg["data"]["image_size_segmentation"]

    x, y = build_segmentation_dataset(raw_dir, masks_dir, image_size=image_size)
    if len(x) == 0:
        raise SystemExit(
            "No segmentation masks found. Run training/generate_sample_data.py or add mask .npy files."
        )

    model = build_bone_landmark_unet(
        input_shape=(image_size, image_size, 1),
        num_classes=cfg["segmentation"]["num_classes"],
    )
    epochs = cfg["segmentation"]["epochs"]
    if os.getenv("MODULE2_QUICK_TRAIN") == "1":
        epochs = min(epochs, 5)

    model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=min(cfg["segmentation"]["batch_size"], len(x)),
        validation_split=0.2,
        verbose=1,
    )

    out_path = Path("training/checkpoints/bone_landmark_unet.keras")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved segmentation model to {out_path}")


if __name__ == "__main__":
    main()
