#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from dataset import build_enamel_segmentation_dataset
from models.enamel_model import build_enamel_unet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    masks_dir = Path(cfg["data"]["processed_dir"]) / "enamel_masks"
    size = cfg["data"]["image_size_enamel"]

    x, y = build_enamel_segmentation_dataset(raw_dir, masks_dir, image_size=size)
    if len(x) == 0:
        raise SystemExit("No enamel masks found. Add .npy masks in training/data/processed/enamel_masks.")

    model = build_enamel_unet(input_shape=(size, size, 1), num_classes=cfg["enamel"]["num_classes"])

    epochs = cfg["enamel"]["epochs"]
    if os.getenv("MODULE1_QUICK_TRAIN") == "1":
        epochs = min(epochs, 5)

    model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=min(cfg["enamel"]["batch_size"], len(x)),
        validation_split=0.2,
        verbose=1,
    )

    out_path = Path("training/checkpoints/enamel_erosion_unet.keras")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved enamel model to {out_path}")


if __name__ == "__main__":
    main()
