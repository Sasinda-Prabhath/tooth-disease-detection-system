from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


def load_image_bytes(data: bytes) -> Tuple[np.ndarray, float | None]:
    """Load PNG/JPG/DICOM bytes into RGB numpy array and optional pixel spacing (mm)."""
    if data[:128].lstrip().startswith(b"DICM") or _looks_like_dicom(data):
        return _load_dicom_bytes(data)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(image), None


def load_image_path(path: Union[str, Path]) -> Tuple[np.ndarray, float | None]:
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return _load_dicom_path(path)
    image = Image.open(path).convert("RGB")
    return np.array(image), None


def _looks_like_dicom(data: bytes) -> bool:
    return pydicom is not None and len(data) > 132 and data[128:132] == b"DICM"


def _load_dicom_bytes(data: bytes) -> Tuple[np.ndarray, float | None]:
    if pydicom is None:
        raise RuntimeError("pydicom is required to read DICOM files.")
    ds = pydicom.dcmread(io.BytesIO(data))
    pixel_spacing = _extract_pixel_spacing(ds)
    arr = ds.pixel_array.astype(np.float32)
    arr = _normalize_to_uint8(arr)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 255 - arr
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr, pixel_spacing


def _load_dicom_path(path: Path) -> Tuple[np.ndarray, float | None]:
    if pydicom is None:
        raise RuntimeError("pydicom is required to read DICOM files.")
    ds = pydicom.dcmread(str(path))
    pixel_spacing = _extract_pixel_spacing(ds)
    arr = ds.pixel_array.astype(np.float32)
    arr = _normalize_to_uint8(arr)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 255 - arr
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr, pixel_spacing


def _extract_pixel_spacing(ds) -> float | None:
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is None:
        return None
    try:
        row, column = float(spacing[0]), float(spacing[1])
        # The public API currently accepts a scalar calibration. Never silently
        # apply row spacing to horizontal distances in anisotropic images.
        if not np.isfinite([row, column]).all() or min(row, column) <= 0 or not np.isclose(row, column, rtol=1e-5):
            return None
        return row
    except (TypeError, IndexError, ValueError):
        return None


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)
