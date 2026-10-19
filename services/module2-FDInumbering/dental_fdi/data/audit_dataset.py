"""Report image counts, per-FDI label coverage and structural dataset problems."""
import argparse
import json
from pathlib import Path

from dental_fdi.analysis.dental_analysis import FDI_CODES
from dental_fdi.data.dataset_checks import check_dataset
from dental_fdi.paths import DATASET_DIR, REPORTS_DIR


def audit(root):
    report = {'problems': check_dataset(root), 'splits': {},
              'limitation': 'Counts and structure do not validate anatomical labels or patient-disjoint splits.'}
    for split in ('train', 'val', 'test'):
        images = [p for p in (root / 'images' / split).glob('*')
                  if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}]
        counts = dict.fromkeys(FDI_CODES, 0)
        for image in images:
            label = root / 'labels' / split / (image.stem + '.txt')
            if not label.is_file():
                continue
            for line in label.read_text().splitlines():
                try:
                    values = list(map(float, line.split()))
                    index = int(values[0])
                    if len(values) == 5 and values[0] == index and 0 <= index < 32:
                        counts[FDI_CODES[index]] += 1
                except (ValueError, IndexError, OverflowError):
                    continue
        report['splits'][split] = {'images': len(images), 'tooth_instances': sum(counts.values()),
                                  'per_fdi': counts, 'classes_without_examples': [c for c in FDI_CODES if counts[c] == 0]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET_DIR)
    parser.add_argument('--output', type=Path, default=REPORTS_DIR / 'dataset_audit.json')
    args = parser.parse_args()
    report = audit(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    for split, stats in report['splits'].items():
        print(f'{split}: {stats["images"]} images, {stats["tooth_instances"]} annotated teeth; FDI counts: {stats["per_fdi"]}')
    print('Problems:', report['problems'])
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
