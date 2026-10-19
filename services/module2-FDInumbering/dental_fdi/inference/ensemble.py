"""One inference pipeline with positive contributions from all three networks."""
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from dental_fdi.analysis.dental_analysis import FDI_CODES
from dental_fdi.inference.hybrid_model import ToothRefiner, crop_and_position

MODEL_NAMES = ('yolo26', 'efficientnet_b0', 'resnet50')
DEFAULT_WEIGHTS = (.5, .25, .25)


def validate_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('All three fusion weights must be finite and strictly positive')
    return weights / weights.sum()


def fuse_rows(rows, efficientnet, resnet, weights=DEFAULT_WEIGHTS):
    """YOLO winning-class evidence + both complete classifier distributions.

    Ultralytics' public detection output provides a winning class/score, not
    the full class posterior. Mix winning-class evidence with a uniform prior as an explicit
    approximation. This is a score ensemble, not calibrated probability fusion.
    """
    weights = validate_weights(weights)
    if len(rows) != len(efficientnet) or len(rows) != len(resnet):
        raise ValueError('Both classifiers must contribute one prediction per detection')
    outputs = []
    for row, eff, res in zip(rows, efficientnet, resnet):
        vectors = []
        for values in (eff, res):
            values = np.asarray(values, dtype=float)
            if (values.shape != (32,) or not np.isfinite(values).all() or (values < 0).any()
                    or not np.isclose(values.sum(), 1, atol=1e-4)):
                raise ValueError('Invalid classifier probability distribution')
            vectors.append(values)
        detector_score = float(row['conf'])
        if not np.isfinite(detector_score) or not 0 <= detector_score <= 1:
            raise ValueError('Invalid YOLO score')
        original = str(row['pred_fdi'])
        original_index = FDI_CODES.index(original)
        evidence = np.full(32, (1 - detector_score) / 32)
        evidence[original_index] += detector_score
        fused = weights[0] * evidence + weights[1] * vectors[0] + weights[2] * vectors[1]
        index = int(fused.argmax())
        final_fdi = FDI_CODES[index]
        # Both classifiers contribute to the score on EVERY proposal, even when
        # the winning FDI stays unchanged. The score never exceeds YOLO's score.
        support = weights[0] + weights[1] * vectors[0][index] + weights[2] * vectors[1][index]
        contributions = {
            'yolo26': {'fdi': original, 'score': detector_score, 'weight': float(weights[0]),
                       'evidence': evidence.tolist()},
            'efficientnet_b0': {'fdi': FDI_CODES[int(vectors[0].argmax())],
                                'score': float(vectors[0].max()), 'weight': float(weights[1]),
                                'probabilities': vectors[0].tolist()},
            'resnet50': {'fdi': FDI_CODES[int(vectors[1].argmax())],
                         'score': float(vectors[1].max()), 'weight': float(weights[2]),
                         'probabilities': vectors[1].tolist()},
        }
        outputs.append(dict(row, pred_fdi=final_fdi, conf=float(detector_score * support),
                            detector_fdi=original, detector_confidence=detector_score,
                            label_refined=final_fdi != original,
                            fusion_class_score=float(fused[index]), fusion_scores=fused.tolist(),
                            model_contributions=contributions,
                            prediction_method='YOLO26 + EfficientNet-B0 + ResNet50'))
    return outputs


def build_ensemble(efficientnet_path, resnet_path, output, weights=DEFAULT_WEIGHTS):
    """Repackage already trained components; never silently initialize weights."""
    weights = validate_weights(weights)
    components, manifests = {}, {}
    detector = None
    for name, source in [('efficientnet_b0', efficientnet_path), ('resnet50', resnet_path)]:
        with zipfile.ZipFile(source) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            if (manifest.get('version') != 1 or manifest.get('classes') != FDI_CODES
                    or manifest.get('architecture') != name
                    or manifest.get('preprocessing') != 'rgb_224_imagenet_pad15_xyxy_v1'):
                raise ValueError(f'Incompatible trained {name} bundle')
            detector_bytes, refiner_bytes = archive.read('detector.pt'), archive.read('refiner.pt')
        for key, content in [('detector', detector_bytes), ('refiner', refiner_bytes)]:
            if hashlib.sha256(content).hexdigest() != manifest[key + '_sha256']:
                raise ValueError(f'Corrupt {name} {key} weights')
        if detector is not None and detector != detector_bytes:
            raise ValueError('Classifier bundles must use identical YOLO26 detector weights')
        detector = detector_bytes
        components[name + '.pt'] = refiner_bytes
        manifests[name] = manifest.get('metrics', {})
    components['detector.pt'] = detector
    manifest = {'version': 2, 'architecture': 'yolo26_efficientnet_b0_resnet50',
                'classes': FDI_CODES, 'models': list(MODEL_NAMES),
                'preprocessing': 'rgb_224_imagenet_pad15_xyxy_v1',
                'fusion': {'method': 'weighted_class_evidence_and_detector_support_v1',
                           'weights': weights.tolist(), 'calibrated': False,
                           'yolo_evidence': 'score * winning one-hot + (1-score) * uniform; not raw class logits'},
                'sha256': {key: hashlib.sha256(value).hexdigest() for key, value in components.items()},
                'component_training': manifests}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        for name, content in components.items():
            archive.writestr(name, content)
    temporary.replace(output)
    return manifest


class ThreeModelFDI:
    def __init__(self, path, device='cpu'):
        from ultralytics import YOLO
        self.device = torch.device(device)
        with zipfile.ZipFile(path) as archive:
            self.manifest = json.loads(archive.read('manifest.json'))
            m = self.manifest
            if (m.get('version') != 2 or m.get('classes') != FDI_CODES or m.get('models') != list(MODEL_NAMES)
                    or m.get('preprocessing') != 'rgb_224_imagenet_pad15_xyxy_v1'
                    or m.get('fusion', {}).get('method') != 'weighted_class_evidence_and_detector_support_v1'):
                raise ValueError('Incompatible three-model schema')
            self.weights = validate_weights(m['fusion']['weights'])
            contents = {name: archive.read(name) for name in ('detector.pt', 'efficientnet_b0.pt', 'resnet50.pt')}
        for name, content in contents.items():
            if hashlib.sha256(content).hexdigest() != m['sha256'][name]:
                raise ValueError(f'Corrupt component: {name}')
        self._directory = tempfile.TemporaryDirectory(prefix='fdi_three_model_')
        detector_path = Path(self._directory.name) / 'detector.pt'
        detector_path.write_bytes(contents['detector.pt'])
        self.detector = YOLO(str(detector_path))
        if (self.detector.task != 'detect' or set(map(str, self.detector.names.values())) != set(FDI_CODES)
                or '26' not in str(self.detector.model.yaml.get('yaml_file', ''))):
            raise ValueError('Expected a dental YOLO26 detector')
        self.classifiers = {}
        for name in MODEL_NAMES[1:]:
            model = ToothRefiner(name)
            model.load_state_dict(torch.load(io.BytesIO(contents[name + '.pt']), map_location='cpu', weights_only=True))
            self.classifiers[name] = model.to(self.device).eval()
        self.names = self.detector.names

    def predict_rows(self, image, confidence, iou, resolution, recovery=False):
        from dental_fdi.inference.detection_inference import infer_detections
        rows = infer_detections(self.detector, image, confidence, iou, resolution, recovery)
        predictions = {name: [] for name in self.classifiers}
        # No classifier bypass or agreement gate: every detected crop runs both.
        with torch.inference_mode():
            for start in range(0, len(rows), 32):
                batch = [crop_and_position(image, row['xyxy']) for row in rows[start:start + 32]]
                crops, positions = zip(*batch)
                crops, positions = torch.stack(crops).to(self.device), torch.stack(positions).to(self.device)
                for name, model in self.classifiers.items():
                    probabilities = model(crops, positions).softmax(1).cpu().tolist()
                    predictions[name].extend(probabilities)
        return fuse_rows(rows, predictions['efficientnet_b0'], predictions['resnet50'], self.weights)


def load_prediction_model(path):
    if str(path).endswith('.fdi.zip'):
        with zipfile.ZipFile(path) as archive:
            version = json.loads(archive.read('manifest.json')).get('version')
        if version == 2:
            return ThreeModelFDI(path)
        if version == 1:
            from dental_fdi.inference.hybrid_model import HybridFDIModel
            return HybridFDIModel(path)
        raise ValueError('Unsupported model bundle version')
    from ultralytics import YOLO
    return YOLO(str(path))
