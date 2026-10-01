"""Train paired crop/geometry rows; unknown labels are excluded, never negatives."""
import argparse
import json
from pathlib import Path
from app.services.impaction import ANGLES, GEOMETRY_VERSION, build_model, image_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rows', type=Path)
    parser.add_argument('--output', type=Path, default=Path('models/impaction/best.pt'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    import mlflow
    torch.manual_seed(20261001)
    rows = json.loads(args.rows.read_text())
    assignments = {}
    for row in rows:
        if assignments.setdefault(row['patient_id'], row['split']) != row['split']:
            raise ValueError('Patient leakage.')
        if row.get('geometry_version') != GEOMETRY_VERSION or len(row['geometry']) != 12:
            raise ValueError('Generate geometry with crop_and_geometry; incompatible feature version.')
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, split):
            self.rows = [r for r in rows if r['split'] == split and type(r.get('impacted')) is bool]
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            r = self.rows[index]
            with Image.open(args.rows.parent/r['crop']) as image:
                x = image_tensor(np.array(image.convert('RGB')))
            return x, torch.tensor(r['geometry'], dtype=torch.float32), int(r['impacted']), ANGLES.index(r['angulation']) if r['impacted'] and r.get('angulation') in ANGLES else -100
    datasets = {s: Dataset(s) for s in ('train', 'val')}
    if any(len(d) == 0 for d in datasets.values()): raise ValueError('Training and validation labels are required.')
    model = build_model().to(args.device)
    # ImageNet initialization is permitted only during training, never serving.
    from torchvision.models import EfficientNet_V2_S_Weights
    state = EfficientNet_V2_S_Weights.DEFAULT.get_state_dict(progress=True)
    model.backbone.load_state_dict({k: v for k, v in state.items() if not k.startswith('classifier.')}, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best = float('inf'); args.output.parent.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run():
        mlflow.log_params({'geometry_version': GEOMETRY_VERSION, 'epochs': args.epochs})
        for epoch in range(args.epochs):
            for split in ('train', 'val'):
                model.train(split == 'train'); total = 0
                for image, geometry, impacted, angle in torch.utils.data.DataLoader(datasets[split], batch_size=8, shuffle=split == 'train'):
                    image, geometry, impacted, angle = [v.to(args.device) for v in (image, geometry, impacted, angle)]
                    with torch.set_grad_enabled(split == 'train'):
                        p, a = model(image, geometry)
                        loss = torch.nn.functional.cross_entropy(p, impacted)
                        valid = angle != -100
                        if valid.any(): loss = loss+torch.nn.functional.cross_entropy(a[valid], angle[valid])
                        if split == 'train':
                            optimizer.zero_grad(); loss.backward(); optimizer.step()
                    total += loss.item()*len(image)
                total /= len(datasets[split]); mlflow.log_metric(split+'_loss', total, step=epoch)
                if split == 'val' and total < best:
                    best = total
                    torch.save({'state_dict': model.cpu().state_dict(), 'geometry_version': GEOMETRY_VERSION, 'angles': ANGLES}, args.output)
                    model.to(args.device)

if __name__ == '__main__': main()
