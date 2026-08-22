from __future__ import annotations


class ModelNotLoadedError(RuntimeError):
    """Raised when inference requires a trained checkpoint but none is available."""

    def __init__(self, model_name: str):
        super().__init__(
            f"Trained model '{model_name}' is not loaded. "
            "Train and export models to models_store/, then restart the API. "
            "Set REQUIRE_TRAINED_MODELS=0 only for development demos."
        )
        self.model_name = model_name


class ImageQualityError(ValueError):
    """Raised when an uploaded image is too small, flat, or blurred for inference."""
