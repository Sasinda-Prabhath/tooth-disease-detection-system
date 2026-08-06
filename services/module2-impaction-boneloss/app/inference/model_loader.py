from __future__ import annotations

from pathlib import Path
from typing import Literal

ModelKind = Literal["keras", "savedmodel"] | None


def _import_tf():
    import tensorflow as tf

    return tf


def _import_keras():
    import keras

    return keras


def resolve_model_path(model_dir: Path) -> tuple[Path | None, ModelKind]:
    """Resolve SavedModel directory or Keras .keras checkpoint path."""
    model_dir = Path(model_dir)

    if (model_dir / "saved_model.pb").exists():
        return model_dir, "savedmodel"

    keras_candidates = [
        model_dir.with_suffix(".keras"),
        model_dir.parent / f"{model_dir.name}.keras",
        model_dir / "model.keras",
    ]
    for candidate in keras_candidates:
        if candidate.exists():
            return candidate, "keras"

    return None, None


def load_inference_model(model_dir: str | Path):
    path, kind = resolve_model_path(Path(model_dir))
    if path is None:
        return None, None

    if kind == "savedmodel":
        tf = _import_tf()
        return tf.saved_model.load(str(path)), "savedmodel"

    keras = _import_keras()
    return keras.models.load_model(str(path), compile=False), "keras"
