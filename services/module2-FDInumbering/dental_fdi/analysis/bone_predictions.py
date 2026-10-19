"""Automatic landmark proposals with explicit failures for each reviewed tooth."""
from dental_fdi.analysis.bone_pose import predict_bone_points
from dental_fdi.analysis.periodontal_measurement import measure_periodontal_sites as measure_bone_sites


def predict_bone_inventory(model, image, boxes, threshold=.5):
    predictions = {}
    for box in boxes:
        entry = {'points': [], 'confidence': [], 'confirmed': False,
                 'revision': 0, 'source': 'Model estimate; requires review'}
        try:
            points, confidence = predict_bone_points(model, image, box['xyxy'], box['fdi'], threshold)
            entry.update(points=points, confidence=confidence)
            if len(points) != 6 or any(p is None for p in points):
                entry['error'] = 'Unavailable: six reliable landmarks were not detected'
            else:
                measure_bone_sites(points, *image.size)
        except Exception as exc:
            entry['error'] = f'Unavailable: {exc}'
        predictions[box['fdi']] = entry
    return predictions
