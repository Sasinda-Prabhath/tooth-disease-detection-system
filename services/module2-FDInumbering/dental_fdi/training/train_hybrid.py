"""Train a frozen ImageNet encoder + FDI/context head and bundle with YOLO26."""
from dental_fdi.paths import PROJECT_ROOT
import argparse
import copy
import json
import random
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from ultralytics import YOLO

from dental_fdi.data.dataset_checks import check_dataset
from dental_fdi.inference.hybrid_model import ToothRefiner, crop_and_position, save_bundle

ROOT = PROJECT_ROOT


@lru_cache(maxsize=2)
def load_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


class ToothCrops(Dataset):
    def __init__(self, root, split):
        self.samples = []
        for path in sorted((root / 'images' / split).glob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                continue
            with Image.open(path) as image:
                if image.getexif().get(274, 1) != 1:
                    raise ValueError(f'{path}: normalize EXIF orientation and update labels before training')
                w, h = image.size
            for line in (root / 'labels' / split / (path.stem + '.txt')).read_text().splitlines():
                if not line.strip():
                    continue
                label, x, y, width, height = map(float, line.split())
                box = [(x - width / 2) * w, (y - height / 2) * h,
                       (x + width / 2) * w, (y + height / 2) * h]
                self.samples.append((path, box, int(label)))
        if not self.samples:
            raise ValueError(f'No labeled teeth in {split}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, box, label = self.samples[index]
        crop, position = crop_and_position(load_rgb(path), box)
        return crop, position, label


def encode(model, dataset, batch, device):
    features, labels = [], []
    model.encoder.eval()
    with torch.inference_mode():
        for index, (crops, positions, target) in enumerate(DataLoader(dataset, batch_size=batch, num_workers=0)):
            encoded = model.encoder(crops.to(device)).cpu()
            features.append(torch.cat((encoded, positions), 1))
            labels.append(target)
            if (index + 1) % 5 == 0:
                print(f'Encoded {sum(len(t) for t in labels)}/{len(dataset)} crops', flush=True)
    # Normal tensors, not inference tensors, are needed by the trainable head.
    return TensorDataset(torch.cat(features).clone(), torch.cat(labels).clone())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--detector', type=Path, default=ROOT / 'runs/detect/train-2/weights/best.pt')
    parser.add_argument('--architecture', choices=['efficientnet_b0', 'resnet50'], default='efficientnet_b0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error('epochs and batch must be positive')
    errors = check_dataset(args.dataset)
    if errors:
        raise ValueError('\n'.join(errors))
    detector = YOLO(str(args.detector))
    from dental_fdi.analysis.dental_analysis import FDI_CODES
    if (detector.task != 'detect' or set(map(str, detector.names.values())) != set(FDI_CODES)
            or '26' not in str(detector.model.yaml.get('yaml_file', ''))):
        raise ValueError('Supply a dental 32-class YOLO26 detection checkpoint')
    # Dataset numeric IDs must have the same order as the classifier and data.yaml.
    if [str(detector.names[i]) for i in range(32)] != FDI_CODES:
        raise ValueError('Detector class order does not match the FDI label schema')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device(args.device)
    model = ToothRefiner(args.architecture, pretrained=True).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    datasets = {split: ToothCrops(args.dataset, split) for split in ('train', 'val')}
    print('Encoding annotated tooth crops:', {k: len(v) for k, v in datasets.items()}, flush=True)
    encoded = {k: encode(model, v, args.batch, device) for k, v in datasets.items()}
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=.01)
    counts = torch.bincount(encoded['train'].tensors[1], minlength=32)
    weights = (counts.sum() / counts.clamp(min=1)).float().sqrt().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=.05)
    best_accuracy, best_loss, best_epoch, best_state, stale = -1, float('inf'), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.head.train()
        for features, labels in DataLoader(encoded['train'], batch_size=args.batch, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.head(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        model.head.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.inference_mode():
            for features, labels in DataLoader(encoded['val'], batch_size=args.batch):
                logits = model.head(features.to(device))
                val_loss += float(torch.nn.functional.cross_entropy(logits, labels.to(device), reduction='sum'))
                correct += int((logits.argmax(1).cpu() == labels).sum())
                total += len(labels)
        accuracy, val_loss = correct / total, val_loss / total
        history.append({'epoch': epoch, 'val_crop_accuracy': accuracy, 'val_loss': val_loss})
        if accuracy > best_accuracy or (accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy, best_loss, best_epoch = accuracy, val_loss, epoch
            best_state = copy.deepcopy(model.head.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: validation crop accuracy {accuracy:.3f}', flush=True)
        if stale >= 25:
            break
    model.head.load_state_dict(best_state)
    metrics = {'best_epoch': best_epoch, 'val_crop_accuracy': best_accuracy,
               'train_crops': len(datasets['train']), 'val_crops': len(datasets['val']),
               'train_class_counts': counts.tolist(), 'encoder_frozen': True,
               'seed': 42, 'history': history,
               'training_arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'limitation': 'Ground-truth crop classification is not end-to-end detection accuracy. Small development split; not independent validation.'}
    output = args.output or ROOT / 'runs/hybrid' / args.architecture / 'candidate.fdi.zip'
    save_bundle(output, args.detector, model, args.architecture, metrics)
    output.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'Candidate bundle saved: {output}. Best crop accuracy: {best_accuracy:.3f}. Not deployed.', flush=True)


if __name__ == '__main__':
    main()
