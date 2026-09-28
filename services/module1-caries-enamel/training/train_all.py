#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs for fast bootstrap")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    py = sys.executable

    env = None
    if args.quick:
        import os

        env = os.environ.copy()
        env["MODULE1_QUICK_TRAIN"] = "1"

    for script in ["train_detector.py", "train_caries.py", "train_enamel.py", "export_savedmodel.py"]:
        run([py, f"training/{script}"], root, env=env)

    print("\nAll Module 1 models trained and exported to models_store/")


if __name__ == "__main__":
    main()
