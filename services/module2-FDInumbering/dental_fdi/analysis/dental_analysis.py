"""Detection postprocessing and reviewed, directed molar-axis measurements."""
import math

FDI_CODES = [f"{q}{t}" for q in range(1, 5) for t in range(1, 9)]
MOLAR_PAIRS = {"18": "17", "28": "27", "38": "37", "48": "47"}
ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11607221/"


def select_detections(rows, width, height, threshold=0.35):
    """Keep strongest per FDI without positional relabeling or cross-class NMS."""
    best = {}
    for row in rows:
        fdi = str(row["pred_fdi"])
        confidence = float(row["conf"])
        coords = list(map(float, row["xyxy"]))
        if fdi not in FDI_CODES or not all(map(math.isfinite, [confidence] + coords)):
            continue
        if confidence < threshold:
            continue
        x1, y1, x2, y2 = coords
        box = [max(0, min(width, x1)), max(0, min(height, y1)),
               max(0, min(width, x2)), max(0, min(height, y2))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if fdi not in best or confidence > best[fdi]["conf"]:
            best[fdi] = dict(row, fdi=fdi, pred_fdi=fdi, xyxy=box,
                             source=row.get('prediction_method', 'Model prediction'))
    return sorted(best.values(), key=lambda row: row["fdi"])


def measure_axes(points, fdi, width, height):
    """Points: third root, third crown, second root, second crown in pixels.

    Standard OPG: patient's right at image left; square pixels. Directed
    root-to-crown vectors preserve inversion. Positive means mesial tilt.
    """
    if fdi not in MOLAR_PAIRS or len(points) != 4:
        raise ValueError("Four axis landmarks and a third-molar FDI code are required.")
    for point in points:
        if point is None or len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
            raise ValueError("Landmarks must contain finite x/y coordinates.")
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError("Landmarks must be inside the image.")
    third = [points[1][i] - points[0][i] for i in (0, 1)]
    second = [points[3][i] - points[2][i] for i in (0, 1)]
    if min(math.hypot(*third), math.hypot(*second)) < 2:
        raise ValueError("Each root-to-crown axis must be at least two pixels long.")
    cross = second[0] * third[1] - second[1] * third[0]
    dot = second[0] * third[0] + second[1] * third[1]
    sign = {"18": -1, "28": 1, "38": -1, "48": 1}[fdi]
    return sign * math.degrees(math.atan2(cross, dot))


def winter_category(angle, fdi):
    """Literal study intervals; fractional gaps require review, not rounding."""
    if not math.isfinite(angle) or not -180 <= angle <= 180:
        raise ValueError("Angle must be finite and between -180 and 180 degrees.")
    if fdi not in MOLAR_PAIRS:
        raise ValueError("Expected a third-molar FDI code.")
    if fdi in ("18", "28"):
        return "Not classified: maxillary tooth outside study scope"
    if -10 <= angle <= 10:
        return "Vertical"
    if 11 <= angle <= 80:
        return "Mesioangular"
    if 81 <= angle <= 100:
        return "Horizontal"
    if -80 <= angle <= -11:
        return "Distoangular"
    return "Outside published intervals / review"


def classify_winter(angle, fdi=None):
    """Clinical Winter's classification for third molars (18, 28, 38, 48).

    Based on Winter's Classification & Patel et al. (2024):
    - Vertical: -10° to 10°
    - Mesioangular: 11° to 80°
    - Horizontal: 81° to 100° (or > 80°)
    - Distoangular: -11° to -80°
    - Inverted / Transverse: > 100° or < -80°
    """
    if angle is None or not math.isfinite(angle):
        return "Unavailable"
    rounded = round(angle)
    if -10 <= rounded <= 10:
        return "Vertical"
    elif 11 <= rounded <= 80:
        return "Mesioangular"
    elif 81 <= rounded <= 105:
        return "Horizontal"
    elif -80 <= rounded <= -11:
        return "Distoangular"
    else:
        return "Inverted / Transverse"


def classify_pell_gregory(third_box, second_box, fdi):
    """Pell & Gregory Depth (A, B, C) and Ramus Space (Class I, II, III).

    Applicable to mandibular third molars (38, 48).
    Reference: Patel H et al. (2024), PMCID: PMC11607221.
    """
    if fdi not in ("38", "48") or not third_box or not second_box:
        return None, None
    x1_3, y1_3, x2_3, y2_3 = third_box
    x1_2, y1_2, x2_2, y2_2 = second_box

    # In mandible: occlusal plane is at top (smaller y)
    y_occ = y1_2
    h2 = max(1.0, y2_2 - y1_2)
    y_neck = y1_2 + 0.38 * h2  # approximate cervical neck line

    # 3rd molar highest occlusal point
    y_top_3 = y1_3
    if y_top_3 <= y_occ + 0.12 * h2:
        depth = "Position A (High: at/above occlusal plane)"
    elif y_top_3 <= y_neck:
        depth = "Position B (Medium: between occlusal plane and neck)"
    else:
        depth = "Position C (Deep: below cervical neck)"

    w3 = max(1.0, x2_3 - x1_3)
    if fdi == "48":
        # 48 is on patient right (image left). Distal is -x, mesial is +x (toward 47)
        overlap = max(0.0, x2_3 - x1_2)
    else:
        # 38 is on patient left (image right). Distal is +x, mesial is -x (toward 37)
        overlap = max(0.0, x2_2 - x1_3)

    if overlap > 0.55 * w3:
        ramus = "Class III (No space / within ramus)"
    elif overlap > 0.15 * w3:
        ramus = "Class II (Reduced space: crown impacted/crowded)"
    else:
        ramus = "Class I (Sufficient space)"

    return depth, ramus


def assess_impaction(angle, winter_cat, depth_cat=None, ramus_cat=None):
    """Clinical impaction determination from angulation, depth, and ramus relationship."""
    if winter_cat in ("Mesioangular", "Horizontal", "Distoangular", "Inverted / Transverse"):
        return f"Impacted ({winter_cat})"
    if depth_cat and ("Position B" in depth_cat or "Position C" in depth_cat):
        pos = depth_cat.split(" ")[0]
        return f"Impacted ({pos})"
    if ramus_cat and ("Class II" in ramus_cat or "Class III" in ramus_cat):
        cls = ramus_cat.split(" ")[0]
        return f"Impacted ({cls})"
    if winter_cat == "Vertical":
        return "Erupted / Normal inclination"
    return "Not determined"


# ---------------------------------------------------------------------------
# Periodontal Bone Loss (PBL) & Severity Analysis
# References:
# - Kao CT et al. (2025/2026), PMCID: PMC11762257 (Patent No. I844494)
# - Periodontal defect morphology & Hamp furcation classification
# ---------------------------------------------------------------------------

def calculate_pbl_ratio(root_length_a, bone_loss_b):
    """Calculate Periodontal Bone Loss ratio R = b / a (CEJ-to-Crest / CEJ-to-Apex)."""
    if not math.isfinite(root_length_a) or root_length_a <= 0:
        return 0.0
    if not math.isfinite(bone_loss_b) or bone_loss_b < 0:
        bone_loss_b = 0.0
    return max(0.0, min(1.0, float(bone_loss_b / root_length_a)))


def classify_pbl_severity(pbl_ratio):
    """Classify PBL severity according to Kao et al. (PMCID: PMC11762257).
    
    - Mild: R <= 1/3 (<= 33.3%) -> Green circle
    - Moderate: 1/3 < R <= 2/3 (33.3% - 66.7%) -> Yellow circle
    - Severe: R > 2/3 (> 66.7%) -> Red circle
    """
    if not math.isfinite(pbl_ratio):
        pbl_ratio = 0.0
    pbl_ratio = max(0.0, min(1.0, float(pbl_ratio)))
    if pbl_ratio <= 0.15:
        return "Normal"
    elif pbl_ratio <= (1.0 / 3.0):
        return "Mild"
    elif pbl_ratio <= (2.0 / 3.0):
        return "Moderate"
    else:
        return "Severe"

