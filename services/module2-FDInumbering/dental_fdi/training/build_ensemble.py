"""Build one three-model artifact from the trained classifier bundles."""
import argparse
from pathlib import Path

from dental_fdi.paths import RUNS_DIR, DEFAULT_ENSEMBLE
from dental_fdi.inference.ensemble import build_ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--efficientnet', type=Path, default=RUNS_DIR / 'hybrid/efficientnet_b0/candidate.fdi.zip')
    parser.add_argument('--resnet', type=Path, default=RUNS_DIR / 'hybrid/resnet50/candidate.fdi.zip')
    parser.add_argument('--output', type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument('--weights', type=float, nargs=3, default=[.5, .25, .25],
                        metavar=('YOLO', 'EFFICIENTNET', 'RESNET'))
    args = parser.parse_args()
    manifest = build_ensemble(args.efficientnet, args.resnet, args.output, args.weights)
    print(f'Created {args.output}; active models: {manifest["models"]}; weights: {manifest["fusion"]["weights"]}')


if __name__ == '__main__':
    main()
