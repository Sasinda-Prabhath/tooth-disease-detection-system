#!/usr/bin/env python3
"""Train molar detector (Stage A) — simplified detector for bootstrap; swap to full YOLOv8 when COCO data is ready."""

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

THIRD_MOLAR_FDI = [18, 28, 38, 48]


def load_training_samples(raw_dir: Path, ann_file: Path, size: int = 640):
    with open(ann_file, encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}

    samples = []

    for ann in coco.get("annotations", []):
        meta = images.get(ann["image_id"])

        if meta is None:
            continue

        img_path = raw_dir / meta["file_name"]

        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size))

        x, y, w, h = ann["bbox"]

        scale_x = size / meta["width"]
        scale_y = size / meta["height"]

        box = np.array(
            [
                x * scale_x,
                y * scale_y,
                (x + w) * scale_x,
                (y + h) * scale_y,
            ],
            dtype=np.float32,
        )

        fdi = ann.get("attributes", {}).get("fdi_number", 18)

        cls = THIRD_MOLAR_FDI.index(fdi) if fdi in THIRD_MOLAR_FDI else 0

        samples.append((image, box, cls))

    return samples


def build_simple_detector(input_shape=(640, 640, 3), num_classes=4):
    import keras

    inputs = keras.Input(shape=input_shape)

    base = keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    x = base(inputs)

    box = keras.layers.Dense(4, name="box")(x)

    cls = keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="class",
    )(x)

    model = keras.Model(inputs, [box, cls])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={
            "box": "mse",
            "class": "sparse_categorical_crossentropy",
        },
        metrics={
            "class": "accuracy",
        },
    )

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/config.yaml"),
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    ann_file = Path(cfg["data"]["annotations_dir"]) / "coco_annotations.json"
    size = cfg["data"]["image_size_detector"]

    samples = load_training_samples(
        raw_dir,
        ann_file,
        size=size,
    )

    if len(samples) == 0:
        raise RuntimeError(
            "No detector samples found. Run training/generate_sample_data.py first."
        )

    xs = np.stack([s[0] for s in samples]).astype(np.float32)

    # Normalize images
    xs /= 255.0

    boxes = np.stack([s[1] for s in samples]).astype(np.float32)

    classes = np.array(
        [s[2] for s in samples],
        dtype=np.int32,
    )

    model = build_simple_detector(
        input_shape=(size, size, 3),
        num_classes=len(THIRD_MOLAR_FDI),
    )

    epochs = cfg["detector"]["epochs"]
    if os.getenv("MODULE2_QUICK_TRAIN") == "1":
        epochs = min(epochs, 5)

    model.fit(
        xs,
        {
            "box": boxes,
            "class": classes,
        },
        epochs=epochs,
        batch_size=8,
        verbose=1,
    )

    # -----------------------------
    # Save model (Keras 3 compatible)
    # -----------------------------
    checkpoint_dir = Path("training/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "molar_detector.keras"

    model.save(str(model_path))

    print("\n======================================")
    print("Detector model saved successfully!")
    print(model_path.resolve())
    print("======================================\n")


if __name__ == "__main__":
    main()