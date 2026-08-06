import io

import numpy as np
from PIL import Image

from app.preprocessing.xray_pipeline import preprocess_panoramic


def test_preprocess_panoramic_png():
    img = Image.new("RGB", (256, 128), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out, spacing = preprocess_panoramic(buf.getvalue())
    assert out.shape == (128, 256, 3)
    assert spacing is None
