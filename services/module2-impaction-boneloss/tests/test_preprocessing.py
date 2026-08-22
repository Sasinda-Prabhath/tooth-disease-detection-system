import io

import numpy as np
from PIL import Image

from app.preprocessing.xray_pipeline import preprocess_panoramic


def test_preprocess_panoramic_png():
    values = np.random.default_rng(7).integers(80, 160, size=(256, 512, 3), dtype=np.uint8)
    img = Image.fromarray(values, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out, spacing = preprocess_panoramic(buf.getvalue())
    assert out.shape == (256, 512, 3)
    assert spacing is None
