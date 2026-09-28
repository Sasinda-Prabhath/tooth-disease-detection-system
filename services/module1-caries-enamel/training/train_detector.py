#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from dataset import build_detector_samples
from models.detector_model import build_caries_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    ann_file = Path(cfg["data"]["annotations_dir"]) / "coco_annotations.json"
    size = cfg["data"]["image_size_detector"]

    x, boxes, classes = build_detector_samples(raw_dir, ann_file, image_size=size)
    if len(x) == 0:
        raise SystemExit("No detector samples found. Add COCO annotations first.")

    model = build_caries_detector(input_shape=(size, size, 3), num_classes=cfg["detector"]["num_classes"])

    epochs = cfg["detector"]["epochs"]
    if os.getenv("MODULE1_QUICK_TRAIN") == "1":
        epochs = min(epochs, 5)

    model.fit(
        x,
        {"box": boxes, "class": classes},
        epochs=epochs,
        batch_size=min(cfg["detector"]["batch_size"], len(x)),
        verbose=1,
    )

    out_path = Path("training/checkpoints/caries_detector.keras")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"Saved detector to {out_path}")


if __name__ == "__main__":
    main()
