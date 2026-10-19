"""Reject known invalid training inputs; passing is not annotation approval."""
from dental_fdi.paths import PROJECT_ROOT
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from dental_fdi.analysis.dental_analysis import FDI_CODES


def check_dataset(root):
    root = Path(root)
    errors = []
    hashes = {}
    splits = ['train', 'val'] + (['test'] if (root / 'images/test').is_dir() else [])
    for split in splits:
        images = [p for p in (root / "images" / split).glob("*")
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]
        hashes[split] = set()
        if not images:
            errors.append(f"{split}: no images")
        for image in images:
            hashes[split].add(hashlib.sha256(image.read_bytes()).hexdigest())
            label = root / "labels" / split / (image.stem + ".txt")
            if not label.exists():
                errors.append(f"{split}/{image.name}: missing annotation")
                continue
            try:
                rows = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
                if not rows:
                    raise ValueError("empty annotation; review explicitly before training")
                seen = set()
                for row in rows:
                    if len(row) != 5:
                        raise ValueError("expected class cx cy width height")
                    if not all(map(math.isfinite, row)):
                        raise ValueError('non-finite annotation')
                    c, x, y, w, h = row
                    if not (c.is_integer() and 0 <= c < 32 and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError("invalid class or coordinates")
                    if c in seen:
                        raise ValueError('duplicate permanent FDI class in the same image')
                    seen.add(c)
                    if not (-1e-5 <= x-w/2 and x+w/2 <= 1+1e-5 and -1e-5 <= y-h/2 and y+h/2 <= 1+1e-5):
                        raise ValueError('tooth box extends outside image')
                presence_file = root / 'presence' / split / (image.stem + '.json')
                if presence_file.is_file():
                    presence = json.loads(presence_file.read_text())
                    statuses = presence.get('status', {})
                    if (presence.get('schema') != 'fdi_presence_annotation_v1' or set(statuses) != set(FDI_CODES)
                            or any(v not in ('present', 'missing', 'uncertain') for v in statuses.values())):
                        raise ValueError('presence annotation must specify present/missing/uncertain for all 32 FDI codes')
                    labeled = {FDI_CODES[int(row[0])] for row in rows}
                    for fdi, status in statuses.items():
                        if status == 'missing' and fdi in labeled:
                            raise ValueError(f'{fdi}: marked missing but has a tooth box')
                        if status == 'present' and fdi not in labeled:
                            raise ValueError(f'{fdi}: marked present but has no tooth box')
                if len(rows) == 32 and all(abs(r[3] - .045) < 1e-6 and abs(r[4] - .18) < 1e-6 and min(abs(r[2] - .42), abs(r[2] - .68)) < 1e-6 for r in rows):
                    errors.append(f"{split}/{image.name}: fixed guessed annotations; replace with verified tooth labels")
            except ValueError as exc:
                errors.append(f"{split}/{label.name}: {exc}")
    for first, second in combinations(splits, 2):
        overlap = hashes[first] & hashes[second]
        if overlap:
            errors.append(f'{first}/{second} leakage: {len(overlap)} unique images occur in both splits')
    return errors


if __name__ == "__main__":
    problems = check_dataset(PROJECT_ROOT / "dataset")
    print("\n".join(problems) if problems else "Structural checks passed. Verify labels and patient separation before training.")
    raise SystemExit(1 if problems else 0)
