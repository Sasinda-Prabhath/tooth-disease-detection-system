"""Two-landmark molar model contract, checks, and automatic angle extraction."""
import hashlib
import json
import math
from pathlib import Path
from dental_fdi.analysis.dental_analysis import MOLAR_PAIRS, measure_axes, winter_category, select_detections

POSE_NAMES = ['17', '18', '27', '28', '37', '38', '47', '48']
AXIS_SCHEMA = {'version': 1, 'names': POSE_NAMES, 'keypoints': ['root_center', 'crown_center'],
               'orientation': 'patient_right_on_image_left', 'kpt_shape': [2, 3]}


def write_axis_schema(weights_dir):
    (Path(weights_dir) / 'axis_schema.json').write_text(json.dumps(AXIS_SCHEMA, indent=2))


def check_pose_dataset(root):
    root = Path(root)
    errors, hashes = [], {}
    for split in ('train', 'val'):
        images = [p for p in (root/'images'/split).glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}]
        hashes[split] = set()
        if not images:
            errors.append(f'{split}: no annotated pose images in {root / "images" / split}')
        for image in images:
            hashes[split].add(hashlib.sha256(image.read_bytes()).hexdigest())
            label = root/'labels'/split/(image.stem+'.txt')
            if not label.is_file():
                errors.append(f'{label}: missing root/crown annotations'); continue
            try:
                lines = label.read_text().splitlines()
                seen = set()
                for line in lines:
                    if not line.strip():
                        continue
                    values = list(map(float, line.split()))
                    if len(values) != 11 or not all(map(math.isfinite, values)):
                        raise ValueError('expected class cx cy w h root_x root_y visibility crown_x crown_y visibility')
                    c,x,y,w,h,rx,ry,rv,cx,cy,cv = values
                    if not c.is_integer() or not 0 <= c < 8 or c in seen:
                        raise ValueError('invalid or duplicate molar class')
                    seen.add(c)
                    if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x-w/2 <= x+w/2 <= 1 and 0 <= y-h/2 <= y+h/2 <= 1):
                        raise ValueError('invalid bounding box')
                    if rv != 2 or cv != 2:
                        raise ValueError('this pipeline requires both root/crown landmarks visibly annotated')
                    if not (x-w/2 <= rx <= x+w/2 and x-w/2 <= cx <= x+w/2 and y-h/2 <= ry <= y+h/2 and y-h/2 <= cy <= y+h/2):
                        raise ValueError('landmarks outside tooth box')
                    if math.hypot(rx-cx, ry-cy) < 1e-6:
                        raise ValueError('degenerate root/crown axis')
            except ValueError as exc:
                errors.append(f'{label}: {exc}')
    if hashes['train'] & hashes['val']:
        errors.append('Duplicate images across pose train/validation splits')
    return errors


def validate_axis_model(model, weights):
    if model.task != 'pose' or [str(model.names[i]) for i in sorted(model.names)] != POSE_NAMES:
        raise ValueError('Expected an eight-class dental molar pose model, not detection or human-pose weights.')
    if list(model.model.model[-1].kpt_shape) != [2, 3]:
        raise ValueError('Expected two root/crown keypoints with confidence.')
    schema_file = Path(weights).parent/'axis_schema.json'
    if not schema_file.is_file() or json.loads(schema_file.read_text()) != AXIS_SCHEMA:
        raise ValueError('Missing or incompatible axis_schema.json beside pose best.pt. Use the supplied pose trainer.')


def pose_rows(result, width, height, box_threshold=.35, keypoint_threshold=.5):
    rows = []
    if result.boxes is None or result.keypoints is None:
        return rows
    for index, box in enumerate(result.boxes):
        points = result.keypoints.data[index].cpu().tolist()
        if len(points) != 2 or any(len(p) != 3 or not all(map(math.isfinite,p)) or p[2] < keypoint_threshold for p in points):
            continue
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        if any(not (0 <= p[0] < width and 0 <= p[1] < height and x1 <= p[0] <= x2 and y1 <= p[1] <= y2) for p in points):
            continue
        if math.dist(points[0][:2],points[1][:2]) < 2:
            continue
        rows.append({'pred_fdi': str(result.names[int(box.cls[0])]), 'conf': float(box.conf[0]),
                     'xyxy': [x1,y1,x2,y2], 'points': [p[:2] for p in points],
                     'keypoint_confidence': [p[2] for p in points]})
    return select_detections(rows, width, height, box_threshold)


def automatic_angles(axes, width, height):
    by_fdi = {row['fdi']: row for row in axes}
    rows = []
    for third, second in MOLAR_PAIRS.items():
        row = {'FDI': third, 'Reference': second, 'Angle (degrees)': None,
               'Classification': 'Unavailable: need both reliable tooth axes',
               'Impaction status': 'Not assessed by this model', 'points': None}
        if third in by_fdi and second in by_fdi:
            points = by_fdi[third]['points'] + by_fdi[second]['points']
            angle = measure_axes(points, third, width, height)
            row.update({'Angle (degrees)': round(angle,2), 'Classification': winter_category(angle,third),
                        'points': points, 'Measurement source': 'Automatic model estimate; unvalidated',
                        'Minimum landmark confidence': min(by_fdi[third]['keypoint_confidence'] + by_fdi[second]['keypoint_confidence'])})
        rows.append(row)
    return rows
