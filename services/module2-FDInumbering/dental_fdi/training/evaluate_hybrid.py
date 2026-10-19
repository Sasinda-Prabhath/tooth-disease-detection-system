"""Compare deployed detector and bundled hybrid at identical operating settings."""
import argparse
import json
import time
from pathlib import Path

from PIL import Image

from dental_fdi.analysis.dental_analysis import FDI_CODES, select_detections
from dental_fdi.inference.detection_inference import infer_detections
from dental_fdi.inference.ensemble import load_prediction_model
from dental_fdi.paths import DATASET_DIR, REPORTS_DIR


def overlap(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def score(predictions, truth):
    matched = set()
    true_positive = 0
    for box in sorted(predictions, key=lambda b: b['conf'], reverse=True):
        candidates = [(overlap(box['xyxy'], target['xyxy']), i) for i, target in enumerate(truth)
                      if i not in matched and target['fdi'] == box['fdi']]
        best_iou, index = max(candidates, default=(0, -1))
        if best_iou >= .5:
            true_positive += 1
            matched.add(index)
    return true_positive, len(predictions) - true_positive, len(truth) - true_positive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hybrid', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DATASET_DIR)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--confidence', type=float, default=.35)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=REPORTS_DIR / 'hybrid_comparison.json')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    hybrid = load_prediction_model(args.hybrid)
    if not hasattr(hybrid, 'detector'):
        raise ValueError('Supply a two-model or three-model bundle')
    models = {'yolo26': hybrid.detector, 'hybrid': hybrid}
    totals = {name: [0, 0, 0] for name in models}
    elapsed = dict.fromkeys(models, 0.)
    details = []
    paths = [p for p in sorted((args.dataset / 'images' / args.split).glob('*'))
             if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}]
    if not paths:
        raise ValueError('No evaluation images')
    for path in paths:
        with Image.open(path) as source:
            if source.getexif().get(274, 1) != 1:
                raise ValueError('Normalize EXIF orientation and labels before evaluation')
            image = source.convert('RGB')
        w, h = image.size
        truth = []
        for line in (args.dataset / 'labels' / args.split / (path.stem + '.txt')).read_text().splitlines():
            if not line.strip():
                continue
            label, x, y, width, height = map(float, line.split())
            truth.append({'fdi': FDI_CODES[int(label)], 'xyxy': [(x-width/2)*w, (y-height/2)*h,
                                                              (x+width/2)*w, (y+height/2)*h]})
        entry = {'image': path.name}
        for name, model in models.items():
            start = time.perf_counter()
            rows = infer_detections(model, image, args.confidence, .5, args.resolution)
            boxes = select_detections(rows, w, h, args.confidence)
            elapsed[name] += time.perf_counter() - start
            values = score(boxes, truth)
            totals[name] = [a + b for a, b in zip(totals[name], values)]
            entry[name] = dict(zip(['tp', 'fp', 'fn'], values))
        details.append(entry)
        print(path.name, entry, flush=True)
    report = {'images': len(paths), 'split': args.split, 'confidence': args.confidence,
              'architecture': hybrid.manifest['architecture'],
              'resolution': args.resolution, 'iou_match': .5, 'per_image': details,
              'limitation': 'Development validation is not independent testing. FDI-aware IoU matching at fixed threshold, not mAP.'}
    for name, (tp, fp, fn) in totals.items():
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        report[name] = {'precision': precision, 'recall': recall,
                        'f1': 2 * precision * recall / max(1e-12, precision + recall),
                        'tp': tp, 'fp': fp, 'fn': fn, 'seconds': elapsed[name]}
    report['hybrid_improves_f1'] = report['hybrid']['f1'] > report['yolo26']['f1']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    args.hybrid.with_suffix('.comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'per_image'}, indent=2))


if __name__ == '__main__':
    main()
