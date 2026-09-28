#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import keras
import tensorflow as tf

CHECKPOINTS = {
    "caries_detector": "training/checkpoints/caries_detector.keras",
    "caries_classifier": "training/checkpoints/caries_classifier.keras",
    "enamel_erosion_unet": "training/checkpoints/enamel_erosion_unet.keras",
}


def export_keras_to_savedmodel(keras_path: Path, savedmodel_path: Path) -> None:
    model = keras.models.load_model(str(keras_path), compile=False)
    if savedmodel_path.exists():
        shutil.rmtree(savedmodel_path)
    savedmodel_path.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(savedmodel_path))
    shutil.copy2(keras_path, savedmodel_path.parent / f"{savedmodel_path.name}.keras")
    print(f"Exported {keras_path.name} -> {savedmodel_path}")


def export_detector(keras_path: Path, savedmodel_path: Path) -> None:
    model = keras.models.load_model(str(keras_path), compile=False)
    if savedmodel_path.exists():
        shutil.rmtree(savedmodel_path)
    savedmodel_path.mkdir(parents=True, exist_ok=True)

    class DetectorServing(tf.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        @tf.function(input_signature=[tf.TensorSpec([None, 640, 640, 3], tf.float32)])
        def __call__(self, x):
            box, cls_probs = self.inner(x, training=False)
            cls_idx = tf.argmax(cls_probs, axis=-1)
            scores = tf.reduce_max(cls_probs, axis=-1)
            return {
                "boxes": box,
                "classes": tf.cast(cls_idx, tf.float32),
                "scores": scores,
            }

    module = DetectorServing(model)
    tf.saved_model.save(module, str(savedmodel_path), signatures={"serving_default": module.__call__})
    shutil.copy2(keras_path, savedmodel_path.parent / f"{savedmodel_path.name}.keras")
    print(f"Exported detector -> {savedmodel_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-store", type=Path, default=Path("models_store"))
    args = parser.parse_args()

    exporters = {
        "caries_detector": export_detector,
        "caries_classifier": export_keras_to_savedmodel,
        "enamel_erosion_unet": export_keras_to_savedmodel,
    }

    args.models_store.mkdir(parents=True, exist_ok=True)

    for name, ckpt_rel in CHECKPOINTS.items():
        ckpt = Path(ckpt_rel)
        out = args.models_store / name
        if not ckpt.exists():
            print(f"Skipping {name}: checkpoint not found at {ckpt}")
            continue
        exporters[name](ckpt, out)


if __name__ == "__main__":
    main()
