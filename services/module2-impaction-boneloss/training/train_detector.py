#!/usr/bin/env python3
"""Train the Module 2 third-molar detector with YOLOv8 on your labelled OPGs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for FDI 18/28/38/48 detection.")
    parser.add_argument("--data", type=Path, default=Path("training/data/yolo/dataset.yaml"), help="YOLO dataset.yaml created by prepare_yolo_dataset.py")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 base checkpoint, e.g. yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None, help="Use 0 for first GPU, cpu for CPU, or omit for auto")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset file not found: {args.data}. Run training/prepare_yolo_dataset.py first.")
    with args.data.open(encoding="utf-8") as file:
        dataset = yaml.safe_load(file)
    if list(dataset.get("names", {}).values()) != ["18", "28", "38", "48"]:
        raise SystemExit("YOLO classes must be exactly: 0=18, 1=28, 2=38, 3=48.")

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs or 100,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="training/runs",
        name="third_molar_yolov8",
        exist_ok=True,
        patience=30,
        pretrained=True,
    )
    _ = results
    best = Path("training/runs/third_molar_yolov8/weights/best.pt")
    if not best.exists():
        raise RuntimeError("YOLO training finished without best.pt; check the training log.")
    destination = Path("models_store/molar_detector/best.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    print(f"YOLOv8 detector exported to {destination.resolve()}")


if __name__ == "__main__":
    main()
