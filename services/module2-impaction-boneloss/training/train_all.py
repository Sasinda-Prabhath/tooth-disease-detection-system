#!/usr/bin/env python3
"""Run full Module 2 training bootstrap: sample data -> train all stages -> export."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parent.parent)


def main() -> None:
    py = sys.executable
    run([py, "training/generate_sample_data.py", "--num-images", "40"])
    run([py, "training/train_detector.py"])
    run([py, "training/train_angulation.py"])
    run([py, "training/train_segmentation.py"])
    run([py, "training/export_savedmodel.py"])
    print("\nAll models trained and exported to models_store/")


if __name__ == "__main__":
    main()
