import math
import os
import cv2
import numpy as np
from app.schemas import ImpactionResult

ANGLES = ['Mesioangular', 'Vertical', 'Horizontal', 'Distoangular']
GEOMETRY_VERSION = 'pca-undirected-v1'

def crop_and_geometry(rgb, tooth, teeth):
    adjacent = next((t for t in teeth if t.fdi == tooth.fdi[0]+'7' and t.status == 'detected'), None)
    boxes = np.array([tooth.box_xyxy] + ([adjacent.box_xyxy] if adjacent else []))
    lo, hi = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    pad = (hi-lo)*.15
    x1, y1 = np.maximum(0, np.floor(lo-pad)).astype(int)
    x2, y2 = np.minimum([rgb.shape[1], rgb.shape[0]], np.ceil(hi+pad)).astype(int)
    def orientation(t):
        pts = np.asarray(t.polygon, dtype=float)
        _, vectors = np.linalg.eigh(np.cov(pts.T))
        v = vectors[:, -1]
        # A polygon does not identify crown versus root: use undirected PCA axis.
        return math.atan2(v[1], v[0]) % math.pi
    angle = orientation(tooth)
    relative = ((angle-orientation(adjacent)+math.pi/2) % math.pi-math.pi/2) if adjacent else 0
    a = np.array(tooth.box_xyxy)
    b = np.array(adjacent.box_xyxy) if adjacent else a
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = (a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection
    distance = np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2) / max(np.linalg.norm(a[2:]-a[:2]), 1)
    features = [float(tooth.fdi == f) for f in ('18', '28', '38', '48')]
    features += [math.sin(2*angle), math.cos(2*angle), math.sin(2*relative), math.cos(2*relative),
                 float(distance), float(intersection/max(union, 1)) if adjacent else 0,
                 float(tooth.fdi[0] in '12'), float(adjacent is not None)]
    return rgb[y1:y2, x1:x2].copy(), (int(x1), int(y1)), np.asarray(features, np.float32)

def image_tensor(rgb):
    import torch
    # This exact transform is shared with training.
    array = cv2.resize(rgb, (384, 384)).astype(np.float32)/255
    array = (array-np.array([.485, .456, .406]))/np.array([.229, .224, .225])
    return torch.from_numpy(array.transpose(2, 0, 1).astype(np.float32))

def build_model():
    import torch
    from torch import nn
    from torchvision.models import efficientnet_v2_s
    class Hybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = efficientnet_v2_s(weights=None)
            self.backbone.classifier = nn.Identity()
            self.geometry = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
            self.impaction = nn.Linear(1280+32, 2)
            self.angulation = nn.Linear(1280+32, 4)
        def forward(self, image, geometry):
            features = torch.cat([self.backbone(image), self.geometry(geometry)], dim=1)
            return self.impaction(features), self.angulation(features)
    return Hybrid()

class ImpactionClassifier:
    def __init__(self, path):
        if not path.is_file():
            raise FileNotFoundError('Trained impaction checkpoint is missing.')
        import torch
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('geometry_version') != GEOMETRY_VERSION or checkpoint.get('angles') != ANGLES:
            raise ValueError('Impaction checkpoint preprocessing/classes are incompatible.')
        self.model = build_model()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.device = torch.device(os.getenv('TORCH_DEVICE', 'cpu'))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, crop, features, fdi):
        import torch
        with torch.inference_mode():
            impaction, angulation = self.model(image_tensor(crop)[None].to(self.device), torch.from_numpy(features)[None].to(self.device))
            p, a = impaction.softmax(1)[0], angulation.softmax(1)[0]
        if not torch.isfinite(p).all() or not torch.isfinite(a).all():
            raise ValueError('Classifier returned non-finite scores.')
        confidence, index = p.max(0)
        if confidence.item() < .7:
            return ImpactionResult(fdi=fdi, impaction_confidence=confidence.item(), reason='Impaction confidence below research threshold (0.70).')
        impacted = bool(index.item())
        return ImpactionResult(fdi=fdi, impacted=impacted, impaction_confidence=confidence.item(),
            angulation=ANGLES[a.argmax().item()] if impacted and a.max().item() >= .7 else None,
            angulation_confidence=a.max().item() if impacted else None,
            reason='Angulation uncertain.' if impacted and a.max().item() < .7 else None)

    def gradcam(self, crop, features):
        import torch
        captured = []
        def capture(_module, _inputs, output):
            output.retain_grad()
            captured.append(output)
        hook = self.model.backbone.features[-1].register_forward_hook(capture)
        try:
            self.model.zero_grad(set_to_none=True)
            logits, _ = self.model(image_tensor(crop)[None].to(self.device), torch.from_numpy(features)[None].to(self.device))
            logits[0, logits.argmax(1).item()].backward()
            activation = captured[0]
            heat = (activation * activation.grad.mean((2, 3), keepdim=True)).sum(1).relu()[0].detach().cpu().numpy()
            heat = cv2.resize(heat/max(float(heat.max()), 1e-8), (crop.shape[1], crop.shape[0]))
            color = cv2.cvtColor(cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(crop, .6, color, .4, 0)
        finally:
            hook.remove()
            self.model.zero_grad(set_to_none=True)
