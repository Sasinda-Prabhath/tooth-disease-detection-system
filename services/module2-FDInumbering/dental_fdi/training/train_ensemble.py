"""Train both classifier branches, then package all three networks together."""
import argparse
import subprocess
import sys
from pathlib import Path
from dental_fdi.inference.ensemble import build_ensemble
from dental_fdi.paths import DATASET_DIR, DEFAULT_DETECTOR, DEFAULT_ENSEMBLE, RUNS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET_DIR)
    parser.add_argument('--detector', type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--output', type=Path, default=DEFAULT_ENSEMBLE)
    args = parser.parse_args()
    bundles = []
    for architecture in ('efficientnet_b0', 'resnet50'):
        bundle = RUNS_DIR / 'hybrid' / architecture / 'candidate.fdi.zip'
        subprocess.run([sys.executable, '-m', 'dental_fdi.training.train_hybrid',
                        '--dataset', str(args.dataset), '--detector', str(args.detector),
                        '--architecture', architecture, '--epochs', str(args.epochs),
                        '--batch', str(args.batch), '--output', str(bundle)], check=True)
        bundles.append(bundle)
    build_ensemble(*bundles, args.output)
    print(f'Three-model artifact: {args.output}')


if __name__ == '__main__':
    main()
