"""YOLO26 + crop/context classifier, distributed as one versioned model bundle.

This is a two-stage model, not a replacement YOLO backbone. Learned crop features
and box position jointly predict FDI; uncertain refinements keep the YOLO label.
"""
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

from dental_fdi.analysis.dental_analysis import FDI_CODES

ARCHITECTURES = ('efficientnet_b0', 'resnet50')
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


class ToothRefiner(nn.Module):
    def __init__(self, architecture='efficientnet_b0', pretrained=False):
        super().__init__()
        if architecture == 'efficientnet_b0':
            self.encoder = models.efficientnet_b0(weights=None)
            if pretrained:
                # Canonical current torchvision weights; older torchvision versions
                # reference a retired hash filename. Always verify the current hash.
                state = torch.hub.load_state_dict_from_url(
                    'https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth',
                    map_location='cpu', check_hash=True)
                self.encoder.load_state_dict(state)
            features = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity()
        elif architecture == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.encoder = models.resnet50(weights=weights)
            features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError(f'Unsupported classifier: {architecture}')
        self.head = nn.Sequential(nn.Linear(features + 4, 256), nn.ReLU(),
                                  nn.Dropout(.2), nn.Linear(256, len(FDI_CODES)))

    def forward(self, crops, positions):
        return self.head(torch.cat((self.encoder(crops), positions), dim=1))


def crop_and_position(image, xyxy):
    """Identical context padding, rounding and coordinates for train/inference."""
    w, h = image.size
    x1, y1, x2, y2 = map(float, xyxy)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Empty tooth crop')
    padx, pady = .15 * (x2 - x1), .15 * (y2 - y1)
    region = (max(0, int(x1 - padx)), max(0, int(y1 - pady)),
              min(w, int(np.ceil(x2 + padx))), min(h, int(np.ceil(y2 + pady))))
    return PREPROCESS(image.convert('RGB').crop(region)), torch.tensor(
        [x1 / w, y1 / h, x2 / w, y2 / h], dtype=torch.float32)


def refine_rows(rows, probabilities, threshold=.85, margin=.25):
    if len(rows) != len(probabilities):
        raise ValueError('Classifier output must match detection count')
    refined = []
    for row, probs in zip(rows, probabilities):
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (32,) or not np.all(np.isfinite(probs)) or np.any(probs < 0) or not np.isclose(probs.sum(), 1, atol=1e-4):
            raise ValueError('Invalid classifier probabilities')
        original = str(row['pred_fdi'])
        index = int(probs.argmax())
        candidate = FDI_CODES[index]
        probability = float(probs[index])
        original_probability = float(probs[FDI_CODES.index(original)])
        changed = candidate != original and probability >= threshold and probability - original_probability >= margin
        refined.append(dict(row, pred_fdi=candidate if changed else original,
                            detector_fdi=original, classifier_fdi=candidate,
                            classifier_confidence=probability, label_refined=bool(changed),
                            # Never boost detection confidence with an uncalibrated classifier.
                            conf=float(row['conf']) * probability if changed else float(row['conf']),
                            prediction_method='YOLO26 + tooth classifier'))
    return refined


def save_bundle(path, detector_path, refiner, architecture, metrics):
    detector_bytes = Path(detector_path).read_bytes()
    manifest = {'version': 1, 'architecture': architecture, 'classes': FDI_CODES,
                'detector_sha256': hashlib.sha256(detector_bytes).hexdigest(),
                'refine_threshold': .85, 'refine_margin': .25,
                'preprocessing': 'rgb_224_imagenet_pad15_xyxy_v1', 'metrics': metrics}
    weights = io.BytesIO()
    torch.save(refiner.cpu().state_dict(), weights)
    manifest['refiner_sha256'] = hashlib.sha256(weights.getvalue()).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        archive.writestr('detector.pt', detector_bytes)
        archive.writestr('refiner.pt', weights.getvalue())
    temporary.replace(path)


class HybridFDIModel:
    def __init__(self, path, device='cpu'):
        from ultralytics import YOLO
        self.device = torch.device(device)
        with zipfile.ZipFile(path) as archive:
            self.manifest = json.loads(archive.read('manifest.json'))
            if (self.manifest['version'] != 1 or self.manifest['classes'] != FDI_CODES or
                    self.manifest['preprocessing'] != 'rgb_224_imagenet_pad15_xyxy_v1'):
                raise ValueError('Incompatible hybrid model schema')
            detector = archive.read('detector.pt')
            refiner = archive.read('refiner.pt')
        for name, value in [('detector', detector), ('refiner', refiner)]:
            if hashlib.sha256(value).hexdigest() != self.manifest[name + '_sha256']:
                raise ValueError(f'Corrupt {name} weights')
        # TemporaryDirectory lives as long as this cached model; no unsafe ZIP extraction.
        self._directory = tempfile.TemporaryDirectory(prefix='fdi_hybrid_')
        weights_path = Path(self._directory.name) / 'detector.pt'
        weights_path.write_bytes(detector)
        self.detector = YOLO(str(weights_path))
        if self.detector.task != 'detect' or set(map(str, self.detector.names.values())) != set(FDI_CODES):
            raise ValueError('Bundle detector must contain the 32 FDI classes')
        if '26' not in str(self.detector.model.yaml.get('yaml_file', '')):
            raise ValueError('Hybrid requires YOLO26 detector weights')
        self.refiner = ToothRefiner(self.manifest['architecture'])
        self.refiner.load_state_dict(torch.load(io.BytesIO(refiner), map_location='cpu', weights_only=True))
        self.refiner.to(self.device).eval()
        self.names = self.detector.names

    def predict_rows(self, image, confidence, iou, resolution, recovery=False):
        from dental_fdi.inference.detection_inference import infer_detections
        rows = infer_detections(self.detector, image, confidence, iou, resolution, recovery)
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(rows), 32):
                tensors = [crop_and_position(image, r['xyxy']) for r in rows[start:start + 32]]
                crops, positions = zip(*tensors)
                logits = self.refiner(torch.stack(crops).to(self.device), torch.stack(positions).to(self.device))
                probabilities.extend(logits.softmax(1).cpu().tolist())
        return refine_rows(rows, probabilities, self.manifest['refine_threshold'], self.manifest['refine_margin'])
