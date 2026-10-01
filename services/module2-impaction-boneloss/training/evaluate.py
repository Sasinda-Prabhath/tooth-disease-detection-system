"""Metrics on dentist-labelled, held-out predictions; missing results stay visible."""
import json
import argparse
from pathlib import Path
import numpy as np

def evaluate(rows):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
    if not rows or any(r.get('split') != 'test' or not r.get('patient_id') for r in rows):
        raise ValueError('Only held-out patient-labelled test rows are accepted.')
    result = {'cases': len(rows)}
    for name in ('impaction', 'angulation'):
        eligible = [r for r in rows if r.get(name+'_true') is not None]
        valid = [r for r in eligible if r.get(name+'_pred') is not None]
        result[name+'_coverage'] = len(valid)/len(eligible) if eligible else None
        if valid:
            truth, pred = [r[name+'_true'] for r in valid], [r[name+'_pred'] for r in valid]
            labels = [False, True] if name == 'impaction' else ['Mesioangular', 'Vertical', 'Horizontal', 'Distoangular']
            p, r, f, _ = precision_recall_fscore_support(truth, pred, average='binary' if name == 'impaction' else 'macro', labels=labels, zero_division=0)
            result[name] = {'precision': float(p), 'recall': float(r), 'f1': float(f), 'accuracy': float(accuracy_score(truth, pred)), 'labels': labels, 'confusion_matrix': confusion_matrix(truth, pred, labels=labels).tolist()}
    valid = [r for r in rows if r.get('bone_loss_true_mm') is not None and r.get('bone_loss_pred_mm') is not None]
    result['bone_loss_pairs'] = len(valid)
    result['bone_loss_mae_mm'] = float(np.mean([abs(r['bone_loss_true_mm']-r['bone_loss_pred_mm']) for r in valid])) if valid else None
    valid = [r for r in rows if r.get('severity_true') is not None and r.get('severity_pred') is not None]
    kappa = float(cohen_kappa_score([r['severity_true'] for r in valid], [r['severity_pred'] for r in valid])) if valid else float('nan')
    result['severity_kappa'] = kappa if np.isfinite(kappa) else None
    return result

def dice(truth, prediction, label):
    if truth.shape != prediction.shape:
        raise ValueError('Segmentation mask dimensions differ.')
    a, b = truth == label, prediction == label
    total = int(a.sum()+b.sum())
    return 2*int((a & b).sum())/total if total else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('rows', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--truth-masks', type=Path)
    parser.add_argument('--predicted-masks', type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.rows.read_text()))
    if bool(args.truth_masks) != bool(args.predicted_masks):
        parser.error('Supply both mask directories.')
    if args.truth_masks:
        from PIL import Image
        files = sorted(args.truth_masks.glob('*.png'))
        if not files: raise ValueError('No held-out masks found.')
        scores = {'CEJ': [], 'crest': []}
        missing = 0
        for truth_file in files:
            predicted_file = args.predicted_masks/truth_file.name
            if not predicted_file.exists():
                missing += 1
                continue
            with Image.open(truth_file) as image: truth = np.array(image)
            with Image.open(predicted_file) as image: prediction = np.array(image)
            if not np.isin(truth, [0,1,2]).all() or not np.isin(prediction, [0,1,2]).all():
                raise ValueError('Mask classes must be 0/1/2.')
            for label, name in ((1, 'CEJ'), (2, 'crest')):
                score = dice(truth, prediction, label)
                if score is not None: scores[name].append(score)
        result['segmentation_coverage'] = (len(files)-missing)/len(files)
        result['dice'] = {name: float(np.mean(values)) if values else None for name, values in scores.items()}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
