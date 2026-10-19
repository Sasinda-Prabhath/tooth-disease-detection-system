"""Unsupervised image-axis detection and molar angulation analysis.

Measures longitudinal molar axes and classifies angulations according to
Winter's Classification and Pell & Gregory criteria (Patel et al., 2024).
"""
import math
import cv2
import numpy as np
from dental_fdi.analysis.dental_analysis import (
    MOLAR_PAIRS,
    measure_axes,
    winter_category,
    classify_winter,
    classify_pell_gregory,
    assess_impaction,
)


def _component_axis(gray, threshold):
    mask = (gray > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    h, w = gray.shape
    candidates = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        coverage = area / (h * w)
        if not 0.08 <= coverage <= 0.88:
            continue
        distance = np.linalg.norm((centers[i] - [w / 2, h / 2]) / [w, h])
        candidates.append((area * max(0.1, 1 - distance), i))
    if not candidates:
        return None
    index = max(candidates)[1]
    ys, xs = np.where(labels == index)
    if len(xs) < 20:
        return None
    cloud = np.column_stack((xs, ys)).astype(float)
    center = cloud.mean(axis=0)
    covariance = np.cov(cloud.T)
    values, vectors = np.linalg.eigh(covariance)
    if values[0] <= 0:
        return None
    elongation = float(values[1] / max(1e-6, values[0]))
    if elongation < 1.12:
        return None
    direction = vectors[:, 1]
    projections = (cloud - center) @ direction
    lo, hi = np.percentile(projections, [7, 93])
    if hi - lo < 6:
        return None
    ends = np.array([center + lo * direction, center + hi * direction])
    contour, _ = cv2.findContours(
        (labels == index).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    c_best = max(contour, key=cv2.contourArea).reshape(-1, 2) if contour else None
    return {
        'ends': ends,
        'direction': direction,
        'elongation': elongation,
        'center': center,
        'contour': c_best,
        'score': float(stats[index, cv2.CC_STAT_AREA]) * max(1.0, elongation),
    }


def estimate_tooth_axis(image, box):
    gray_image = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    x1, y1, x2, y2 = [int(round(v)) for v in box['xyxy']]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    w_box, h_box = x2 - x1, y2 - y1
    if w_box < 10 or h_box < 10:
        return None, 'Tooth crop too small'

    gray = gray_image[y1:y2, x1:x2]
    if np.percentile(gray, 95) - np.percentile(gray, 5) < 12:
        return None, 'Insufficient contrast'

    # Normalize contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    fdi = box['fdi']
    is_upper = fdi[0] in '12'

    otsu, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proposals = []
    # Test range of intensity levels
    t_levels = [otsu - 20, otsu - 10, otsu, otsu + 10, otsu + 20]
    for t in t_levels:
        t_clamped = int(np.clip(t, 25, 235))
        p = _component_axis(blurred, t_clamped)
        if p is not None:
            proposals.append(p)

    # Top-percentile moment directions
    for p_pct in [60, 70, 80]:
        t_p = np.percentile(blurred, p_pct)
        mask_p = (blurred > t_p).astype(np.uint8)
        M = cv2.moments(mask_p)
        if M['m00'] > 20 and (M['mu20'] + M['mu02']) > 0:
            ang = 0.5 * math.atan2(2 * M['mu11'], M['mu20'] - M['mu02']) + math.pi / 2
            dir_vec = np.array([math.cos(ang), math.sin(ang)])
            cen = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
            proposals.append({
                'ends': np.array([cen - 0.4 * max(w_box, h_box) * dir_vec, cen + 0.4 * max(w_box, h_box) * dir_vec]),
                'direction': dir_vec,
                'elongation': 1.25,
                'center': cen,
                'contour': None,
                'score': float(M['m00']),
            })

    if not proposals:
        cen = np.array([w_box / 2.0, h_box / 2.0])
        dir_vec = np.array([1.0, 0.0]) if w_box > h_box * 1.2 else np.array([0.0, 1.0])
        proposals.append({
            'ends': np.array([cen - 0.4 * max(w_box, h_box) * dir_vec, cen + 0.4 * max(w_box, h_box) * dir_vec]),
            'direction': dir_vec,
            'elongation': 1.15,
            'center': cen,
            'contour': None,
            'score': 1.0,
        })

    # Weighted circular consensus of directions (lines: symmetry modulo pi)
    angles_2 = [2 * math.atan2(p['direction'][1], p['direction'][0]) for p in proposals]
    weights = [max(1.0, p.get('score', 1.0)) for p in proposals]
    sin_sum = sum(w * math.sin(a) for a, w in zip(angles_2, weights))
    cos_sum = sum(w * math.cos(a) for a, w in zip(angles_2, weights))
    consensus_theta = 0.5 * math.atan2(sin_sum, cos_sum)
    consensus_dir = np.array([math.cos(consensus_theta), math.sin(consensus_theta)])

    # Calculate angular deviations among valid proposals
    deviations = []
    for p in proposals:
        dot = abs(float(np.clip(np.dot(consensus_dir, p['direction']), -1.0, 1.0)))
        deviations.append(math.degrees(math.acos(dot)))
    # Report standard deviation or 90th percentile deviation
    axis_spread = float(np.std(deviations)) if len(deviations) > 1 else 0.0

    best_p = max(proposals, key=lambda p: p.get('score', 1.0))
    center = best_p['center']
    half_len = max(w_box, h_box) * 0.42
    p1 = center - half_len * consensus_dir
    p2 = center + half_len * consensus_dir

    p1_img = [float(p1[0] + x1), float(p1[1] + y1)]
    p2_img = [float(p2[0] + x1), float(p2[1] + y1)]

    # Crown vs Root polarity determination:
    # Second molars (17, 27, 37, 47) are erupted reference teeth:
    # Upper molars: root in maxilla (smaller y), crown occlusal (larger y)
    # Lower molars: root in mandible (larger y), crown occlusal (smaller y)
    if fdi in ('17', '27'):
        root, crown = (p1_img, p2_img) if p1_img[1] < p2_img[1] else (p2_img, p1_img)
    elif fdi in ('37', '47'):
        root, crown = (p1_img, p2_img) if p1_img[1] > p2_img[1] else (p2_img, p1_img)
    else:
        # Third molars (18, 28, 38, 48):
        is_horizontal = (abs(consensus_dir[0]) > abs(consensus_dir[1]) * 1.15) or (w_box > h_box * 1.15)
        if is_horizontal:
            # Horizontal third molar:
            # 48 & 18: adjacent second molar is to the right (+x), so crown is on the right (+x)
            # 38 & 28: adjacent second molar is to the left (-x), so crown is on the left (-x)
            if fdi in ('48', '18'):
                root, crown = (p1_img, p2_img) if p1_img[0] < p2_img[0] else (p2_img, p1_img)
            else:
                root, crown = (p1_img, p2_img) if p1_img[0] > p2_img[0] else (p2_img, p1_img)
        else:
            # Vertical or tilted third molar:
            if is_upper:
                root, crown = (p1_img, p2_img) if p1_img[1] < p2_img[1] else (p2_img, p1_img)
            else:
                root, crown = (p1_img, p2_img) if p1_img[1] > p2_img[1] else (p2_img, p1_img)

    for pt in (root, crown):
        pt[0] = float(np.clip(pt[0], x1, x2 - 1))
        pt[1] = float(np.clip(pt[1], y1, y2 - 1))

    contour_list = (best_p['contour'] + [x1, y1]).tolist() if best_p.get('contour') is not None else None

    return {
        'fdi': fdi,
        'points': [root, crown],
        'axis_spread_degrees': round(axis_spread, 2),
        'contour': contour_list,
        'direction_assumption': 'Longitudinal axis based on multi-scale segmentation consensus',
    }, None


def estimate_image_angles(image, boxes):
    axes, reasons = {}, {}
    box_map = {b['fdi']: b for b in boxes}
    required = set(MOLAR_PAIRS) | set(MOLAR_PAIRS.values())

    for box in boxes:
        if box['fdi'] in required:
            proposal, reason = estimate_tooth_axis(image, box)
            if proposal:
                axes[box['fdi']] = proposal
            else:
                reasons[box['fdi']] = reason

    rows = []
    for third, second in MOLAR_PAIRS.items():
        row = {
            'FDI': third,
            'Reference': second,
            'Estimated angle (degrees)': None,
            'Angle (degrees)': None,
            'Winter Classification': 'Unavailable',
            'Provisional category': 'Unavailable',
            'Method': "Winter's longitudinal axis method (Patel et al. 2024)",
            'Status': 'Needs reliable detection and image axes for both teeth',
            'Impaction': 'Not determined from angulation alone',
            'Pell & Gregory': None,
            'points': None,
        }

        if third in box_map:
            if second in box_map:
                # If reference axis had slight segmentation trouble, provide anatomical vertical fallback
                if second not in axes:
                    b_sec = box_map[second]
                    x1_s, y1_s, x2_s, y2_s = b_sec['xyxy']
                    cx_s = (x1_s + x2_s) / 2.0
                    is_upp = second[0] in '12'
                    r_sec = [cx_s, y1_s + 0.1 * (y2_s - y1_s)] if is_upp else [cx_s, y2_s - 0.1 * (y2_s - y1_s)]
                    c_sec = [cx_s, y2_s - 0.1 * (y2_s - y1_s)] if is_upp else [cx_s, y1_s + 0.1 * (y2_s - y1_s)]
                    axes[second] = {'fdi': second, 'points': [r_sec, c_sec], 'axis_spread_degrees': 0.0}

                if third in axes and second in axes:
                    points = axes[third]['points'] + axes[second]['points']
                    signed = measure_axes(points, third, image.width, image.height)
                    w_cat = classify_winter(signed, third)
                    depth, ramus = classify_pell_gregory(box_map[third]['xyxy'], box_map[second]['xyxy'], third)
                    impaction = assess_impaction(signed, w_cat, depth, ramus)

                    spread = max(
                        axes[third].get('axis_spread_degrees', 0.0),
                        axes[second].get('axis_spread_degrees', 0.0)
                    )

                    row.update({
                        'Estimated angle (degrees)': round(signed, 1),
                        'Angle (degrees)': round(signed, 1),
                        'Winter Classification': w_cat,
                        'Provisional category': w_cat,
                        'Pell & Gregory': f"{depth}; {ramus}" if depth else "N/A (Maxillary tooth)",
                        'Impaction': impaction,
                        'Status': 'Measured from longitudinal axes',
                        'points': points,
                        'Threshold axis spread (degrees)': round(spread, 1),
                    })
                else:
                    row['Status'] = '; '.join(f'{fdi}: {reasons.get(fdi, "could not establish axis")}' for fdi in (third, second) if fdi not in axes)
            else:
                row['Status'] = f'{second}: not detected'
                row['Impaction'] = 'Reference tooth missing'
        else:
            row['Status'] = f'{third}: not detected'
            row['Provisional category'] = 'Absent / Not detected'
            row['Impaction'] = 'Tooth not detected in arch'

        rows.append(row)
    return rows, axes
