#!/usr/bin/env python3
"""Run full Module 2 training bootstrap: sample data -> train all stages -> export."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=40)
    parser.add_argument("--skip-data", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs for fast bootstrap")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    py = sys.executable

    if not args.skip_data:
        run([py, "training/generate_sample_data.py", "--num-images", str(args.num_images)], root)

    env = None
    if args.quick:
        import os

        env = os.environ.copy()
        env["MODULE2_QUICK_TRAIN"] = "1"

    scripts = [
        "train_tooth_instance.py",
        "prepare_yolo_dataset.py",
        "train_detector.py",
        "train_angulation.py",
        "train_segmentation.py",
        "export_savedmodel.py",
    ]
    import os

    if os.getenv("MODULE2_SKIP_INSTANCE") == "1":
        scripts = [s for s in scripts if s != "train_tooth_instance.py"]

    for script in scripts:
        cmd = [py, f"training/{script}"]
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.check_call(cmd, cwd=root, env=env)

    print("\nAll models trained and exported to models_store/")


if __name__ == "__main__":
    main()
