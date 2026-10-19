"""Legacy heuristic prototype, retained for historical tests; not used by the current measurement UI.

Use periodontal_review and manual_geometry for reviewed landmarks, or a trained
six-landmark model satisfying bone_pose.BONE_SCHEMA. Outputs below are not
validated clinical predictions.

Implements clinical radiographic analysis based on:
1. Kao CT, Tseng CJ, Tsai MC, Huang CN (2025/2026).
   "Determination of periodontal bone loss in panoramic radiographs through artificial intelligence analysis".
   Journal of Dental Sciences / PMCID: PMC11762257.
   - Formula: PBL Ratio R = b / a (Bone Loss depth b / Total Root length a).
   - Severity Grading:
     * Mild: R <= 1/3 (<= 33.3%) -> Green indicator
     * Moderate: 1/3 < R <= 2/3 (33.3% - 66.7%) -> Yellow indicator
     * Severe: R > 2/3 (> 66.7%) -> Red indicator
   - Traffic light indicators for clear clinical and patient communication.

2. Diagnostic Accuracy of CBCT and Intraoral Radiographs for Bone Morphology & Furcation:
   - Defect morphology: Horizontal (Suprabony) vs. Angular (Vertical / Infrabony).
   - Infrabony depth (AC-BD): Shallow (< 3 mm) vs. Deep (>= 3 mm).
   - Furcation involvement in pluriradicular elements (Hamp et al.): Class 0, I, II, III.
   - Clinical 2017 AAP/EFP Periodontitis Staging (Stage I to Stage IV).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# Multi-rooted molars susceptible to furcation involvement
MOLAR_FDIS = {"16", "17", "18", "26", "27", "28", "36", "37", "38", "46", "47", "48"}

# Standard color palette for medical traffic-light indicators
TRAFFIC_COLORS = {
    "Normal": {
        "hex": "#38ef7d",
        "rgb": (56, 239, 125),
        "bgr": (125, 239, 56),
        "emoji": "🟢",
        "label": "Healthy / Intact",
    },
    "Mild": {
        "hex": "#2ea043",
        "rgb": (46, 160, 67),
        "bgr": (67, 160, 46),
        "emoji": "🟢",
        "label": "Mild (≤ 1/3)",
    },
    "Moderate": {
        "hex": "#e3b341",
        "rgb": (227, 179, 65),
        "bgr": (65, 179, 227),
        "emoji": "🟡",
        "label": "Moderate (1/3 - 2/3)",
    },
    "Severe": {
        "hex": "#f85149",
        "rgb": (248, 81, 73),
        "bgr": (73, 81, 248),
        "emoji": "🔴",
        "label": "Severe (> 2/3)",
    },
}


def calculate_pbl_ratio(root_length_a: float, bone_loss_b: float) -> float:
    """Calculate the Periodontal Bone Loss Ratio R = b / a.

    Args:
        root_length_a: Distance from CEJ to apex (a). Must be > 0.
        bone_loss_b: Distance from CEJ to marginal alveolar bone crest (b).

    Returns:
        float: PBL ratio R in range [0.0, 1.0].
    """
    if not math.isfinite(root_length_a) or root_length_a <= 0:
        return 0.0
    if not math.isfinite(bone_loss_b) or bone_loss_b < 0:
        bone_loss_b = 0.0
    ratio = bone_loss_b / root_length_a
    return max(0.0, min(1.0, float(ratio)))


def classify_pbl_severity(pbl_ratio: float) -> Dict[str, Any]:
    """Classify bone loss severity according to Kao et al. (PMCID: PMC11762257).

    - Mild: PBL <= 1/3 (<= 33.3%) -> Green circle
    - Moderate: 1/3 < PBL <= 2/3 (33.3% - 66.7%) -> Yellow circle
    - Severe: PBL > 2/3 (> 66.7%) -> Red circle

    Returns:
        dict with severity grade, color, traffic light emoji, and AAP/EFP stage.
    """
    if not math.isfinite(pbl_ratio):
        pbl_ratio = 0.0
    pbl_ratio = max(0.0, min(1.0, float(pbl_ratio)))
    pct = pbl_ratio * 100.0

    one_third = 1.0 / 3.0
    two_thirds = 2.0 / 3.0

    if pbl_ratio <= 0.15:
        grade = "Normal"
        meta = TRAFFIC_COLORS["Normal"]
        stage = "Stage 0 (Clinically Intact / Healthy)"
    elif pbl_ratio <= one_third:
        grade = "Mild"
        meta = TRAFFIC_COLORS["Mild"]
        stage = "Stage I (Coronal 1/3, < 15-33% root length)"
    elif pbl_ratio <= two_thirds:
        grade = "Moderate"
        meta = TRAFFIC_COLORS["Moderate"]
        stage = "Stage II / III (Middle 1/3, 33-66% root length)"
    else:
        grade = "Severe"
        meta = TRAFFIC_COLORS["Severe"]
        stage = "Stage III / IV (Apical 1/3, > 66% root length)"

    return {
        "grade": grade,
        "ratio": pbl_ratio,
        "percent": pct,
        "label": meta["label"],
        "emoji": meta["emoji"],
        "hex": meta["hex"],
        "rgb": meta["rgb"],
        "bgr": meta["bgr"],
        "stage": stage,
    }


def classify_furcation_involvement(fdi: str, pbl_ratio: float) -> Dict[str, Any]:
    """Determine furcation involvement according to Hamp et al. (1975) classification.

    Applies to pluriradicular teeth (upper and lower molars).
    """
    if fdi not in MOLAR_FDIS:
        return {
            "involved": False,
            "class_name": "N/A (Single-Rooted)",
            "grade": 0,
            "description": "Single-rooted tooth; furcation assessment not applicable.",
        }

    # Anatomical root trunk: furcation entrance typically sits at ~30-35% root length
    if pbl_ratio < 0.30:
        return {
            "involved": False,
            "class_name": "Class 0 (No Furcation)",
            "grade": 0,
            "description": "Alveolar bone remains coronal to the furcation entrance.",
        }
    elif pbl_ratio < 0.45:
        return {
            "involved": True,
            "class_name": "Class I (Initial Furcation)",
            "grade": 1,
            "description": "Horizontal bone loss < 3 mm into the furcation entrance.",
        }
    elif pbl_ratio < 0.60:
        return {
            "involved": True,
            "class_name": "Class II (Partial Furcation)",
            "grade": 2,
            "description": "Horizontal bone loss > 3 mm without through-and-through penetration.",
        }
    else:
        return {
            "involved": True,
            "class_name": "Class III (Through-and-Through)",
            "grade": 3,
            "description": "Complete through-and-through furcation involvement.",
        }


def extract_tooth_landmarks_and_pbl(
    image_np: np.ndarray,
    box: List[float],
    fdi: str,
    mm_per_pixel: float = 0.12,
) -> Dict[str, Any]:
    """Detect anatomical landmarks (Apex, CEJ, Alveolar Crest, Base of Defect) and compute PBL.

    Args:
        image_np: Grayscale or BGR numpy array of full panoramic radiograph.
        box: Bounding box [x1, y1, x2, y2].
        fdi: FDI two-digit code (e.g. '16', '46').
        mm_per_pixel: Estimated millimeter resolution per pixel.

    Returns:
        dict with coordinates, distances (px & mm), PBL ratio, morphology, and furcation.
    """
    h_img, w_img = image_np.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    box_w = max(4, x2 - x1)
    box_h = max(6, y2 - y1)
    cx = (x1 + x2) // 2

    is_upper = str(fdi).startswith(("1", "2"))

    # Convert to grayscale for gradient/density analysis
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np

    crop = gray[y1:y2, x1:x2]

    # Anatomical orientation:
    # Upper arch (Maxilla): Root apex at top (near y1), Crown at bottom (near y2).
    # Lower arch (Mandible): Crown at top (near y1), Root apex at bottom (near y2).
    if is_upper:
        # Occlusal surface near bottom, Apex near top
        occlusal_y = y2 - int(0.05 * box_h)
        apex_y = y1 + int(0.05 * box_h)
        anatomical_crown_h = 0.32 * box_h
        default_cej_y = y2 - int(anatomical_crown_h)
    else:
        # Occlusal surface near top, Apex near bottom
        occlusal_y = y1 + int(0.05 * box_h)
        apex_y = y2 - int(0.05 * box_h)
        anatomical_crown_h = 0.32 * box_h
        default_cej_y = y1 + int(anatomical_crown_h)

    # Refine CEJ and Alveolar Crest using image intensity & horizontal gradient profiles
    refined_cej_y = default_cej_y
    refined_ac_y = None
    mesial_ac_y = None
    distal_ac_y = None

    if crop.size > 0 and box_h > 15:
        # Vertical profile: column-wise average intensity in central 50% width
        sub_w1 = int(0.25 * box_w)
        sub_w2 = int(0.75 * box_w)
        center_strip = crop[:, sub_w1:sub_w2]
        if center_strip.size > 0:
            vert_profile = np.mean(center_strip, axis=1)

            # Smooth profile
            kernel_size = max(3, (box_h // 10) | 1)
            vert_smooth = cv2.GaussianBlur(vert_profile.astype(np.float32), (1, kernel_size), 0).ravel()
            grad = np.gradient(vert_smooth)

            # Search around expected CEJ
            cej_rel = default_cej_y - y1
            search_min = max(2, int(cej_rel - 0.15 * box_h))
            search_max = min(box_h - 2, int(cej_rel + 0.15 * box_h))

            if search_max > search_min:
                local_grad = grad[search_min:search_max]
                best_idx = np.argmax(np.abs(local_grad)) + search_min
                refined_cej_y = y1 + best_idx

        # Mesial and Distal bone crest search in lateral interdental margins
        # Left interdental border (first 25% width) and Right border (last 25% width)
        left_strip = crop[:, :max(2, int(0.22 * box_w))]
        right_strip = crop[:, max(0, int(0.78 * box_w)):]

        def find_interproximal_crest(strip: np.ndarray, is_upper_arch: bool, cej_rel_y: int) -> int:
            if strip.size == 0:
                return cej_rel_y
            prof = np.mean(strip, axis=1)
            prof_smooth = cv2.GaussianBlur(prof.astype(np.float32), (1, max(3, (box_h // 12) | 1)), 0).ravel()
            g = np.gradient(prof_smooth)

            if is_upper_arch:
                # Searching towards apex (smaller y)
                s_start = max(1, int(0.10 * box_h))
                s_end = max(s_start + 1, min(box_h - 1, cej_rel_y))
                if s_end > s_start:
                    crest_rel = s_start + np.argmax(np.abs(g[s_start:s_end]))
                    return crest_rel
            else:
                # Mandible: searching towards apex (larger y)
                s_start = max(1, cej_rel_y)
                s_end = min(box_h - 1, int(0.90 * box_h))
                if s_end > s_start:
                    crest_rel = s_start + np.argmax(np.abs(g[s_start:s_end]))
                    return crest_rel
            return cej_rel_y

        cej_rel_crop = max(0, min(box_h - 1, refined_cej_y - y1))
        crest_left = find_interproximal_crest(left_strip, is_upper, cej_rel_crop)
        crest_right = find_interproximal_crest(right_strip, is_upper, cej_rel_crop)

        # Map to quadrant (mesial vs distal):
        # Q1/Q4: patient right (image left is distal, image right is mesial towards midline)
        # Q2/Q3: patient left (image left is mesial towards midline, image right is distal)
        quadrant = str(fdi)[0] if len(str(fdi)) == 2 else "1"
        if quadrant in ("1", "4"):
            distal_ac_y = y1 + crest_left
            mesial_ac_y = y1 + crest_right
        else:
            mesial_ac_y = y1 + crest_left
            distal_ac_y = y1 + crest_right

        # Average crest level
        refined_ac_y = int(round((mesial_ac_y + distal_ac_y) / 2.0))

    # Physiological normal baseline margin between CEJ and alveolar crest (~1.5 mm, ~12 px)
    phys_margin_px = max(2.0, 1.5 / mm_per_pixel)

    # Compute lengths:
    # a: Total root length = |Apex - CEJ|
    root_len_px = abs(apex_y - refined_cej_y)
    root_len_px = max(root_len_px, 10.0)

    # b: Bone loss depth = |AC - CEJ| - physiological margin
    if refined_ac_y is None:
        raw_b_px = 0.0
    else:
        raw_b_px = abs(refined_ac_y - refined_cej_y)

    bone_loss_px = max(0.0, raw_b_px - phys_margin_px)

    # Compute ratio R = b / a
    pbl_ratio = calculate_pbl_ratio(root_len_px, bone_loss_px)
    severity_info = classify_pbl_severity(pbl_ratio)

    # Defect Morphology Assessment:
    # Compare mesial vs distal crest levels (AC-BD component)
    if mesial_ac_y is not None and distal_ac_y is not None:
        crest_diff_px = abs(mesial_ac_y - distal_ac_y)
        crest_diff_mm = crest_diff_px * mm_per_pixel
    else:
        crest_diff_px = 0.0
        crest_diff_mm = 0.0

    infrabony_threshold_mm = 2.0
    if crest_diff_mm >= infrabony_threshold_mm and pbl_ratio > 0.15:
        morphology = "Angular / Infrabony (Vertical)"
        infrabony_depth_mm = crest_diff_mm
        infrabony_class = "Deep (≥ 3 mm)" if infrabony_depth_mm >= 3.0 else "Shallow (< 3 mm)"
        is_angular = True
    else:
        morphology = "Horizontal (Suprabony)"
        infrabony_depth_mm = 0.0
        infrabony_class = "None / Suprabony"
        is_angular = False

    # Furcation involvement for molars
    furcation_info = classify_furcation_involvement(fdi, pbl_ratio)

    # Convert to estimated mm
    root_len_mm = root_len_px * mm_per_pixel
    bone_loss_mm = bone_loss_px * mm_per_pixel

    return {
        "fdi": str(fdi),
        "is_upper": is_upper,
        "box": [x1, y1, x2, y2],
        "center_x": cx,
        "landmarks": {
            "apex": (cx, apex_y),
            "cej": (cx, refined_cej_y),
            "alveolar_crest": (cx, refined_ac_y if refined_ac_y is not None else refined_cej_y),
            "mesial_crest": (x1 if quadrant in ("2", "3") else x2, mesial_ac_y or refined_cej_y),
            "distal_crest": (x2 if quadrant in ("2", "3") else x1, distal_ac_y or refined_cej_y),
        },
        "lengths": {
            "root_length_px": round(root_len_px, 1),
            "bone_loss_px": round(bone_loss_px, 1),
            "root_length_mm": round(root_len_mm, 1),
            "bone_loss_mm": round(bone_loss_mm, 1),
        },
        "pbl_ratio": round(pbl_ratio, 4),
        "pbl_percent": round(severity_info["percent"], 1),
        "severity": severity_info["grade"],
        "severity_label": severity_info["label"],
        "severity_emoji": severity_info["emoji"],
        "severity_hex": severity_info["hex"],
        "severity_rgb": severity_info["rgb"],
        "aap_efp_stage": severity_info["stage"],
        "defect_morphology": morphology,
        "is_angular": is_angular,
        "infrabony_depth_mm": round(infrabony_depth_mm, 1),
        "infrabony_class": infrabony_class,
        "furcation": furcation_info,
    }


def analyze_periodontal_bone_loss(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    mm_per_pixel: float = 0.12,
) -> Dict[str, Any]:
    """Analyze all detected teeth for Periodontal Bone Loss (PBL) and clinical defect staging.

    Args:
        image: PIL Image of the panoramic radiograph.
        detections: List of detection dicts with 'fdi' and 'xyxy'.
        mm_per_pixel: Scale calibration (approx 0.12 mm/pixel for standard 2000px OPG).

    Returns:
        dict containing 'teeth' results list, 'summary' metrics, and 'global_stage'.
    """
    image_np = np.array(image.convert("RGB"))
    teeth_results: List[Dict[str, Any]] = []

    for det in detections:
        fdi = str(det.get("fdi", det.get("pred_fdi", "")))
        box = det.get("xyxy")
        if not fdi or not box or len(box) != 4:
            continue

        result = extract_tooth_landmarks_and_pbl(image_np, box, fdi, mm_per_pixel=mm_per_pixel)
        result["confidence"] = det.get("conf")
        result["source"] = det.get("source", "Model")
        teeth_results.append(result)

    # Sort results by FDI code
    teeth_results.sort(key=lambda t: t["fdi"])

    # Aggregate clinical summary
    total = len(teeth_results)
    normal_count = sum(1 for t in teeth_results if t["severity"] == "Normal")
    mild_count = sum(1 for t in teeth_results if t["severity"] == "Mild")
    moderate_count = sum(1 for t in teeth_results if t["severity"] == "Moderate")
    severe_count = sum(1 for t in teeth_results if t["severity"] == "Severe")
    angular_count = sum(1 for t in teeth_results if t["is_angular"])
    furcation_count = sum(1 for t in teeth_results if t["furcation"]["involved"])

    max_pbl_tooth = None
    max_pbl_ratio = 0.0
    for t in teeth_results:
        if t["pbl_ratio"] > max_pbl_ratio:
            max_pbl_ratio = t["pbl_ratio"]
            max_pbl_tooth = t["fdi"]

    # Global AAP/EFP Patient Staging
    if severe_count >= 2 or max_pbl_ratio > 0.667:
        patient_stage = "Stage IV / Advanced Periodontitis"
        stage_desc = "Severe bone loss (> 66% root length) or multiple teeth with deep alveolar destruction."
        traffic_stage = "Severe"
    elif moderate_count >= 2 or max_pbl_ratio > 0.333:
        patient_stage = "Stage II / III Periodontitis"
        stage_desc = "Moderate bone loss (33% to 66% root length), localized or generalized."
        traffic_stage = "Moderate"
    elif mild_count >= 1 or max_pbl_ratio > 0.15:
        patient_stage = "Stage I Periodontitis (Mild)"
        stage_desc = "Mild bone loss limited to coronal third (<= 33% root length)."
        traffic_stage = "Mild"
    else:
        patient_stage = "Periodontally Healthy / Gingivitis"
        stage_desc = "Physiological bone levels maintained across all detected teeth."
        traffic_stage = "Normal"

    return {
        "teeth": teeth_results,
        "summary": {
            "total_teeth": total,
            "normal_count": normal_count,
            "mild_count": mild_count,
            "moderate_count": moderate_count,
            "severe_count": severe_count,
            "angular_defect_count": angular_count,
            "furcation_involved_count": furcation_count,
            "max_pbl_ratio": round(max_pbl_ratio, 4),
            "max_pbl_percent": round(max_pbl_ratio * 100.0, 1),
            "max_pbl_tooth": max_pbl_tooth,
            "patient_stage": patient_stage,
            "patient_stage_desc": stage_desc,
            "traffic_stage": traffic_stage,
        },
    }


def render_periodontal_overlay(
    image: Image.Image,
    pbl_analysis: Dict[str, Any],
    show_cej: bool = True,
    show_ac: bool = True,
    show_apex: bool = True,
    show_traffic_lights: bool = True,
    show_badges: bool = True,
) -> Image.Image:
    """Render medical overlay of Periodontal Bone Loss onto panoramic radiograph.

    Follows the Kao et al. (2025/2026) patent & clinical publication style:
    - Red dashed line: Cementoenamel junction (CEJ)
    - Blue dashed line: Alveolar bone crest line (AC)
    - Green dashed line: Apical root line (Apex)
    - Traffic light circles (🟢 Green = Mild, 🟡 Yellow = Moderate, 🔴 Red = Severe)
    - FDI and PBL percentage badges.
    """
    overlay_img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay_img)
    w_img, h_img = image.size

    try:
        font_badge = ImageFont.truetype("arialbd.ttf", 22)
        font_sub = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        try:
            font_badge = ImageFont.truetype("arial.ttf", 20)
            font_sub = ImageFont.load_default()
        except Exception:
            font_badge = ImageFont.load_default()
            font_sub = ImageFont.load_default()

    teeth = pbl_analysis.get("teeth", [])
    if not teeth:
        return overlay_img.convert("RGB")

    # Group teeth by upper and lower arches for continuous contour lines
    upper_teeth = [t for t in teeth if t["is_upper"]]
    lower_teeth = [t for t in teeth if not t["is_upper"]]

    def draw_dashed_polyline(points: List[Tuple[int, int]], color: Tuple[int, int, int, int], width: int = 3, dash_len: int = 10, gap_len: int = 6):
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            num_dashes = int(dist / (dash_len + gap_len))
            for d in range(num_dashes + 1):
                t_start = (d * (dash_len + gap_len)) / dist
                t_end = min(1.0, ((d * (dash_len + gap_len)) + dash_len) / dist)
                if t_start >= 1.0:
                    break
                sx = int(p1[0] + dx * t_start)
                sy = int(p1[1] + dy * t_start)
                ex = int(p1[0] + dx * t_end)
                ey = int(p1[1] + dy * t_end)
                draw.line([(sx, sy), (ex, ey)], fill=color, width=width)

    # Draw continuous anatomical landmark lines for each arch
    for arch_list in (upper_teeth, lower_teeth):
        if not arch_list:
            continue
        # Sort left to right
        sorted_arch = sorted(arch_list, key=lambda t: t["center_x"])

        # Apex line (Green dashed)
        if show_apex and len(sorted_arch) >= 2:
            apex_pts = [t["landmarks"]["apex"] for t in sorted_arch]
            draw_dashed_polyline(apex_pts, color=(56, 239, 125, 220), width=3)

        # CEJ line (Red/Pink dashed)
        if show_cej and len(sorted_arch) >= 2:
            cej_pts = [t["landmarks"]["cej"] for t in sorted_arch]
            draw_dashed_polyline(cej_pts, color=(255, 75, 120, 240), width=3)

        # Alveolar Crest line (Blue/Cyan dashed)
        if show_ac and len(sorted_arch) >= 2:
            ac_pts = [t["landmarks"]["alveolar_crest"] for t in sorted_arch]
            draw_dashed_polyline(ac_pts, color=(0, 210, 255, 240), width=3)

    # Draw per-tooth annotations
    for idx, t in enumerate(teeth):
        box = t["box"]
        x1, y1, x2, y2 = box
        cx = t["center_x"]
        sev_rgb = t["severity_rgb"]
        color_rgba = (sev_rgb[0], sev_rgb[1], sev_rgb[2], 255)

        cej_pt = t["landmarks"]["cej"]
        ac_pt = t["landmarks"]["alveolar_crest"]
        apex_pt = t["landmarks"]["apex"]

        # Draw measurement line b (CEJ to AC) and line a (CEJ to Apex)
        # Line a: faint white guide
        draw.line([cej_pt, apex_pt], fill=(220, 220, 220, 160), width=2)
        # Line b: colored bone loss indicator
        draw.line([cej_pt, ac_pt], fill=color_rgba, width=4)

        # Draw Traffic Light Circle on crown/cervical area (matching Kao et al. Fig. 1)
        if show_traffic_lights:
            radius = max(9, min(16, (x2 - x1) // 4))
            if t["is_upper"]:
                circle_cy = y2 - int(0.18 * (y2 - y1))
            else:
                circle_cy = y1 + int(0.18 * (y2 - y1))

            circle_bbox = [cx - radius, circle_cy - radius, cx + radius, circle_cy + radius]
            # Outer dark border
            draw.ellipse([cx - radius - 2, circle_cy - radius - 2, cx + radius + 2, circle_cy + radius + 2], fill=(13, 17, 23, 230))
            # Filled traffic light circle
            draw.ellipse(circle_bbox, fill=color_rgba, outline=(255, 255, 255, 240), width=2)

        # Draw Badges
        if show_badges:
            fdi_str = f"#{t['fdi']}"
            pbl_str = f"{t['pbl_percent']}%"
            badge_title = f"{fdi_str} {pbl_str}"

            try:
                bbox_t = font_badge.getbbox(badge_title)
                tw = bbox_t[2] - bbox_t[0]
                th = bbox_t[3] - bbox_t[1]
            except Exception:
                tw, th = len(badge_title) * 11, 20

            # Stagger badges to avoid overlap
            stagger = (idx % 2)
            if t["is_upper"]:
                b_y1 = y1 - (42 if stagger == 0 else 76)
                b_y1 = max(b_y1, 8)
            else:
                b_y1 = y2 + (8 if stagger == 0 else 44)
                b_y1 = min(b_y1, h_img - th - 20)

            b_x1 = max(8, min(w_img - tw - 24, cx - tw // 2))
            b_x2 = b_x1 + tw + 16
            b_y2 = b_y1 + th + 10

            # Badge background with severity border
            draw.rectangle([b_x1, b_y1, b_x2, b_y2], fill=(16, 21, 29, 235), outline=color_rgba, width=2)
            draw.text((b_x1 + 8, b_y1 + 4), badge_title, font=font_badge, fill=(255, 255, 255, 255))

    return overlay_img.convert("RGB")
