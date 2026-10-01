import json
import math
import os
import numpy as np
from app.schemas import BoneResult, Site

def measure(fdi, sites, spacing):
    for site in sites:
        site.bone_loss_mm = None
    if spacing is None:
        return BoneResult(fdi=fdi, sites=sites, reason='Pixel spacing is unavailable; accurate millimetre measurement cannot be calculated.')
    if len(spacing) != 2 or not all(math.isfinite(s) and s > 0 for s in spacing):
        raise ValueError('Pixel spacing must contain positive finite row and column values.')
    values = []
    for site in sites:
        if site.cej_point is None or site.crest_point is None:
            continue
        dx = (site.crest_point[0]-site.cej_point[0])*spacing[1]
        dy = (site.crest_point[1]-site.cej_point[1])*spacing[0]
        site.bone_loss_mm = round(math.hypot(dx, dy), 4)
        values.append(site.bone_loss_mm)
    return BoneResult(fdi=fdi, sites=sites, assessable=bool(values), mean_bone_loss_mm=round(float(np.mean(values)), 4) if values else None,
        reason=None if values else 'CEJ/crest landmarks could not be paired reliably.')

def landmarks_from_mask(mask, tooth, origin):
    # Extract per-surface points near the target tooth, not the adjacent molar.
    # Heuristic requires dentist validation; abstain when components are ambiguous.
    import cv2
    x0, y0 = origin
    box = np.array(tooth.box_xyxy)-np.array([x0, y0, x0, y0])
    width, height = box[2]-box[0], box[3]-box[1]
    # Image-left/right surface attribution is unreliable for horizontal/tilted molars.
    points = np.asarray(tooth.polygon, dtype=float)
    _, axes = np.linalg.eigh(np.cov(points.T))
    if abs(axes[1, -1]) < .85:
        return [Site(side='mesial'), Site(side='distal')]
    center = (box[0]+box[2])/2
    sites = []
    for side in ('mesial', 'distal'):
        right = (side == 'mesial') == (tooth.fdi[0] in '14')
        selected = []
        for label in (1, 2):
            ys, xs = np.indices(mask.shape)
            band = (xs >= (center if right else box[0]-.2*width)) & (xs <= (box[2]+.2*width if right else center))
            band &= (ys >= box[1]) & (ys <= box[3])
            n, components, stats, centroids = cv2.connectedComponentsWithStats(((mask == label) & band).astype(np.uint8))
            candidates = [centroids[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 3]
            selected.append(tuple((candidates[0]+[x0, y0]).tolist()) if len(candidates) == 1 else None)
        sites.append(Site(side=side, cej_point=selected[0], crest_point=selected[1]))
    return sites

class BoneSegmenter:
    def __init__(self, root):
        if not (root/'fold_0'/'checkpoint_final.pth').is_file():
            raise FileNotFoundError('Trained nnU-Net 2D fold_0 checkpoint is missing.')
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        metadata = json.loads((root/'dataset.json').read_text())
        if metadata.get('labels') != {'background': 0, 'CEJ': 1, 'crest': 2} or len(metadata.get('channel_names', {})) != 1:
            raise ValueError('nnU-Net dataset must have one grayscale channel and background/CEJ/crest classes.')
        self.model = nnUNetPredictor(device=torch.device(os.getenv('TORCH_DEVICE', 'cpu')), perform_everything_on_device=False, verbose=False)
        self.model.initialize_from_trained_model_folder(str(root), use_folds=(0,), checkpoint_name='checkpoint_final.pth')
        if len(self.model.configuration_manager.patch_size) != 2:
            raise ValueError('Expected nnU-Net 2D configuration.')

    def predict(self, crop, tooth, origin, spacing):
        import cv2
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Fixed training pixel-grid spacing; physical measurement uses original coordinates separately.
        mask = self.model.predict_single_npy_array(gray[None, None], {'spacing': [999, 1, 1]}, None, None, False)[0]
        if mask.shape != gray.shape or not np.isin(mask, [0, 1, 2]).all():
            raise ValueError('Invalid nnU-Net segmentation shape or class values.')
        return measure(tooth.fdi, landmarks_from_mask(mask, tooth, origin), spacing)
