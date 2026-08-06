#!/usr/bin/env python3
"""Export Keras checkpoints to TensorFlow SavedModel for inference API."""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import tensorflow as tf


def export_keras_to_savedmodel(keras_path: Path, savedmodel_path: Path) -> None:
    model = keras.models.load_model(keras_path, compile=False)
    savedmodel_path.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(model, str(savedmodel_path))
    print(f"Exported {keras_path.name} -> {savedmodel_path}")


def export_detector(keras_path: Path, savedmodel_path: Path) -> None:
    """Export detector with inference signature expected by app.inference.detector."""
    model = keras.models.load_model(keras_path, compile=False)
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
            fdi_map = tf.constant([18, 28, 38, 48], dtype=tf.int32)
            fdi = tf.gather(fdi_map, cls_idx)
            return {
                "boxes": box,
                "classes": tf.cast(fdi - 1, tf.float32),
                "scores": scores,
            }

    module = DetectorServing(model)
    tf.saved_model.save(module, str(savedmodel_path), signatures={"serving_default": module.__call__})
    print(f"Exported detector -> {savedmodel_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-store", type=Path, default=Path("models_store"))
    args = parser.parse_args()

    checkpoints = {
        "molar_detector": ("training/checkpoints/molar_detector", export_detector),
        "angulation_classifier": ("training/checkpoints/angulation_classifier", export_keras_to_savedmodel),
        "bone_landmark_unet": ("training/checkpoints/bone_landmark_unet", export_keras_to_savedmodel),
    }

    for name, (ckpt_rel, exporter) in checkpoints.items():
        ckpt = Path(ckpt_rel)
        out = args.models_store / name
        if not ckpt.exists():
            print(f"Skipping {name}: checkpoint not found at {ckpt}")
            continue
        if name == "molar_detector":
            exporter(ckpt, out)
        else:
            out.mkdir(parents=True, exist_ok=True)
            exporter(ckpt, out)


if __name__ == "__main__":
    main()
