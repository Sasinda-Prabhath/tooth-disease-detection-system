#!/usr/bin/env python3
"""Train angulation classifier (Stage B)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_angulation_dataset
from models.angulation_model import build_angulation_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    ann_file = Path(cfg["data"]["annotations_dir"]) / "coco_annotations.json"
    image_size = cfg["data"]["image_size_angulation"]

    x, y = build_angulation_dataset(raw_dir, ann_file, image_size=image_size)
    if len(x) == 0:
        raise SystemExit(
            "No angulation training samples found. Run training/generate_sample_data.py or add COCO annotations."
        )

    model = build_angulation_classifier(
        input_shape=(image_size, image_size, 3),
        num_classes=cfg["angulation"]["num_classes"],
    )
    epochs = cfg["angulation"]["epochs"]
    if os.getenv("MODULE2_QUICK_TRAIN") == "1":
        epochs = min(epochs, 5)

    model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=min(cfg["angulation"]["batch_size"], len(x)),
        validation_split=0.2,
        verbose=1,
    )

    out_path = Path("training/checkpoints/angulation_classifier.keras")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved angulation classifier to {out_path}")


if __name__ == "__main__":
    main()
