from io import BytesIO
import math
import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_PIXELS = 24_000_000

def decode(payload: bytes, filename: str, manual_spacing: float | None = None):
    spacing, source = None, None
    if filename.lower().endswith(('.dcm', '.dicom')) or payload[128:132] == b'DICM':
        import pydicom
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
        ds = pydicom.dcmread(BytesIO(payload))
        if int(ds.Rows) * int(ds.Columns) > MAX_PIXELS or int(getattr(ds, 'NumberOfFrames', 1)) != 1:
            raise ValueError('Only single-frame OPGs up to 24 megapixels are supported.')
        if getattr(ds, 'PhotometricInterpretation', '') not in ('MONOCHROME1', 'MONOCHROME2'):
            raise ValueError('DICOM must be monochrome.')
        pixels = apply_voi_lut(apply_modality_lut(ds.pixel_array, ds), ds).astype(np.float32)
        if not np.isfinite(pixels).all() or pixels.max() == pixels.min():
            raise ValueError('DICOM has no usable pixel contrast.')
        gray = cv2.normalize(pixels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            gray = 255 - gray
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if hasattr(ds, 'PixelSpacing'):
            spacing = tuple(float(x) for x in ds.PixelSpacing)
            source = 'DICOM PixelSpacing (row, column); acquisition calibration requires review'
    else:
        with Image.open(BytesIO(payload)) as im:
            if im.format not in ('PNG', 'JPEG') or im.width * im.height > MAX_PIXELS:
                raise ValueError('Use PNG, JPEG or single-frame DICOM up to 24 megapixels.')
            rgb = np.array(ImageOps.exif_transpose(im).convert('RGB'))
    if manual_spacing is not None:
        if not math.isfinite(manual_spacing) or manual_spacing <= 0:
            raise ValueError('Verified pixel spacing must be finite and positive.')
        spacing, source = (manual_spacing, manual_spacing), 'User-supplied verified isotropic calibration'
    if spacing is not None and (len(spacing) != 2 or not all(math.isfinite(s) and s > 0 for s in spacing)):
        raise ValueError('Invalid DICOM PixelSpacing.')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if rgb.shape[1] < 320 or rgb.shape[0] < 160 or gray.std() < 4:
        raise ValueError('OPG fails quality check: minimum 320 × 160 pixels and grayscale standard deviation 4.0.')
    return rgb, spacing, source

def preprocess(rgb, privacy):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask, warnings = privacy.mask(rgb)
    gray[mask > 0] = 0
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray[mask > 0] = 0  # Enhancement must not restore masked pixels.
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), warnings
