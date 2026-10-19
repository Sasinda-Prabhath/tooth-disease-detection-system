"""Dental Panoramic Radiograph (OPG) Enhancement Pipeline.

Provides medical-grade radiological preprocessing for dental X-rays:
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Unsharp Masking (High-frequency enamel & root sharpening)
- Adaptive Gamma & Percentile Dynamic Range Normalization
- Bilateral Edge-Preserving Denoising
- Teeth Isolation Masking (Teeth Only, Pure Black Background)
"""
import cv2
import numpy as np
from PIL import Image


def isolate_teeth_only(
    image: Image.Image,
    background_dim: float = 0.0,
    clip_limit: float = 3.0,
    sharpen_strength: float = 1.4
) -> Image.Image:
    """Isolate dental teeth structures onto a black background.

    Suppresses cervical spine shadows, skull base, air cavities, and mandible
    borders, keeping only the teeth (enamel crowns, dentin, and roots) illuminated.
    """
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 1. CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Focus on anatomical dental arch corridor (approx. 20% to 88% vertical height)
    dental_band = np.zeros_like(gray, dtype=np.uint8)
    dental_band[int(0.20 * h):int(0.88 * h), int(0.04 * w):int(0.96 * w)] = 255

    # 3. Multi-level Otsu + localized density thresholding for tooth structures
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dental structures threshold
    thresh_val = max(50, otsu_val - 18)
    mask_dense = ((blurred > thresh_val) & (dental_band > 0)).astype(np.uint8) * 255

    # 4. Morphological dilation & closing to bridge roots and crowns
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask_dense, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # 5. Connected component filtering to remove small background speckles
    cnt, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean_mask = np.zeros_like(mask)
    for i in range(1, cnt):
        if stats[i, cv2.CC_STAT_AREA] >= 180:
            clean_mask[labels == i] = 255

    # Smooth edges with light dilation
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.dilate(clean_mask, k_dilate, iterations=1)

    # 6. Unsharp sharpen the tooth pixels
    if sharpen_strength > 0:
        gauss = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharp = cv2.addWeighted(enhanced, 1.0 + (sharpen_strength * 0.5), gauss, -(sharpen_strength * 0.5), 0)
    else:
        sharp = enhanced

    # 7. Apply black background (or dimmed background)
    bg = (gray * background_dim).astype(np.uint8)
    result = np.where(clean_mask > 0, sharp, bg)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))


def isolate_detected_teeth(
    image: Image.Image,
    boxes: list,
    padding: int = 6,
    background_dim: float = 0.0
) -> Image.Image:
    """Isolate all detected teeth boxes on a pure black background."""
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    gauss = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharp = cv2.addWeighted(enhanced, 1.4, gauss, -0.4, 0)

    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        mask[y1:y2, x1:x2] = 255

    bg = (gray * background_dim).astype(np.uint8)
    result = np.where(mask > 0, sharp, bg)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))


def enhance_dental_xray(
    image: Image.Image,
    preset: str = "clahe_sharp",
    clip_limit: float = 2.5,
    tile_grid_size: int = 8,
    sharpen_strength: float = 1.3,
    gamma: float = 1.0,
    denoise: bool = False,
    background_dim: float = 0.0
) -> Image.Image:
    """Enhance a panoramic dental radiograph for AI detection and clinical interpretation.

    Presets:
    - 'clahe_sharp': Balanced CLAHE + unsharp masking for sharp crowns and roots (Recommended).
    - 'teeth_isolated_black': Isolates teeth only; masks all non-dental background to black.
    - 'root_penetration': High CLAHE + gamma correction to illuminate deep ramus / wisdom tooth regions.
    - 'crown_detail': High-frequency unsharp mask to accentuate enamel, restorations, and occlusal tables.
    - 'denoise_clahe': Bilateral edge-preserving filtering + CLAHE for grainy scans.
    - 'custom': User-configured clip_limit, sharpen_strength, and gamma.
    """
    if preset == "teeth_isolated_black":
        return isolate_teeth_only(image, background_dim=background_dim, clip_limit=clip_limit, sharpen_strength=sharpen_strength)

    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # 1. Preset configuration
    if preset == "clahe_sharp":
        clip_limit = 2.5
        tile_grid_size = 8
        sharpen_strength = 1.3
        gamma = 0.95
        denoise = False
    elif preset == "root_penetration":
        clip_limit = 3.5
        tile_grid_size = 12
        sharpen_strength = 1.1
        gamma = 0.85  # Lift dark mandibular ramus and tuberosity shadows
        denoise = False
    elif preset == "crown_detail":
        clip_limit = 2.0
        tile_grid_size = 8
        sharpen_strength = 1.8
        gamma = 1.0
        denoise = False
    elif preset == "denoise_clahe":
        clip_limit = 2.5
        tile_grid_size = 8
        sharpen_strength = 1.0
        gamma = 1.0
        denoise = True

    # 2. Edge-preserving Denoising (Bilateral filter)
    if denoise:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)

    # 3. Dynamic Range Percentile Normalization (removes extreme sensor saturation/borders)
    p_low, p_high = np.percentile(gray, [0.5, 99.5])
    if p_high > p_low:
        stretched = np.clip((gray.astype(float) - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
    else:
        stretched = gray

    # 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(stretched)

    # 5. Adaptive Gamma Correction (lifts alveolar bone shadows)
    if abs(gamma - 1.0) > 0.01 and gamma > 0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in np.arange(0, 256)]).astype(np.uint8)
        cl = cv2.LUT(cl, table)

    # 6. Unsharp Masking (High-frequency detail enhancement)
    if sharpen_strength > 0:
        gaussian = cv2.GaussianBlur(cl, (0, 0), sigmaX=2.0, sigmaY=2.0)
        alpha = sharpen_strength * 0.5
        unsharp = cv2.addWeighted(cl, 1.0 + alpha, gaussian, -alpha, 0)
        result_gray = np.clip(unsharp, 0, 255).astype(np.uint8)
    else:
        result_gray = cl

    return Image.fromarray(cv2.cvtColor(result_gray, cv2.COLOR_GRAY2RGB))


def get_enhancement_presets():
    return {
        "Balanced contrast (default)": "clahe_sharp",
        "Teeth Only (Isolated on Black Background)": "teeth_isolated_black",
        "Root detail": "root_penetration",
        "Crown detail": "crown_detail",
        "Reduced noise": "denoise_clahe",
        "Custom adjustments": "custom"
    }
