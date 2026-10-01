import cv2
import os
import numpy as np
from app.schemas import FDI, Tooth

def validate_teeth(teeth):
    best = {}
    for tooth in teeth:
        if tooth.fdi not in best or tooth.confidence > best[tooth.fdi].confidence:
            best[tooth.fdi] = tooth
    teeth = list(best.values())
    def flag(t, reason):
        t.status = 'uncertain'
        if reason not in t.warnings:
            t.warnings.append(reason)
    for t in teeth:
        if t.confidence < .5:
            flag(t, 'Low detection confidence.')
    center = lambda t: ((t.box_xyxy[0]+t.box_xyxy[2])/2, (t.box_xyxy[1]+t.box_xyxy[3])/2)
    upper = [t for t in teeth if t.fdi[0] in '12']
    lower = [t for t in teeth if t.fdi[0] in '34']
    if upper and lower:
        boundary = (np.median([center(t)[1] for t in upper])+np.median([center(t)[1] for t in lower]))/2
        for t in teeth:
            if (t.fdi[0] in '12' and center(t)[1] >= boundary) or (t.fdi[0] in '34' and center(t)[1] <= boundary):
                flag(t, 'Upper/lower jaw position is inconsistent.')
    # Radiological convention: patient right appears on image left. Never relabel.
    for jaw in (upper, lower):
        ordered = sorted(jaw, key=lambda t: center(t)[0])
        ranks = lambda t: 8-int(t.fdi[1]) if t.fdi[0] in '14' else 7+int(t.fdi[1])
        for a, b in zip(ordered, ordered[1:]):
            if ranks(a) >= ranks(b):
                flag(a, 'FDI sequence/orientation needs review.')
                flag(b, 'FDI sequence/orientation needs review.')
    return sorted(teeth, key=lambda t: t.fdi)

class FDIDetector:
    def __init__(self, weights):
        if not weights.is_file():
            raise FileNotFoundError('Trained FDI segmentation weights are missing.')
        from ultralytics import YOLO
        self.model = YOLO(str(weights), task='segment')
        if self.model.task != 'segment' or [str(self.model.names[i]) for i in range(len(self.model.names))] != FDI:
            raise ValueError('Expected segmentation weights with the ordered 32 FDI classes; COCO weights are invalid.')

    def predict(self, rgb):
        result = self.model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf=.15, verbose=False, device=os.getenv('TORCH_DEVICE', 'cpu'))[0]
        if len(result.boxes) and result.masks is None:
            raise ValueError('FDI model returned no tooth polygons.')
        teeth = []
        h, w = rgb.shape[:2]
        for i, box in enumerate(result.boxes):
            coords = box.xyxy[0].cpu().numpy().astype(float)
            coords[[0, 2]] = np.clip(coords[[0, 2]], 0, w-1)
            coords[[1, 3]] = np.clip(coords[[1, 3]], 0, h-1)
            polygon = result.masks.xy[i].astype(float)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w-1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h-1)
            if len(polygon) < 3 or not np.isfinite(polygon).all() or coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError('Invalid tooth geometry from FDI model.')
            teeth.append(Tooth(fdi=FDI[int(box.cls.item())], box_xyxy=coords.tolist(), polygon=polygon.tolist(), confidence=float(box.conf.item())))
        return validate_teeth(teeth)
