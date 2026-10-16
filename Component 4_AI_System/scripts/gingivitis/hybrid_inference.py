from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from torchvision import models, transforms
from ultralytics import YOLO


PROJECT_ROOT = Path(r"D:\Gingivities Dataset")
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "gingivitis_yolo26s_final_best.pt"
EFFICIENTNET_MODEL_PATH = (
    PROJECT_ROOT / "models" / "efficientnet_b0_stage2_severity_best.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "hybrid_outputs"

YOLO_CONFIDENCE_THRESHOLD = 0.25
CLASS_NAMES = {
    0: "Mild",
    1: "Moderate",
    2: "Severe",
}

EFFICIENTNET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid YOLO26s + EfficientNet-B0 gingivitis inference."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to one full dental image.",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate_file(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unreadable image file: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not open image file: {path}") from exc


def build_efficientnet_b0(num_classes: int = 3) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("EfficientNet checkpoint does not contain a valid state_dict.")
    return state_dict


def load_efficientnet(path: Path, device: torch.device) -> nn.Module:
    model = build_efficientnet_b0(num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_yolo(path: Path) -> YOLO:
    return YOLO(str(path))


def safe_output_path(image_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / f"{image_path.stem}_hybrid.jpg"
    if not base.exists():
        return base

    index = 2
    while True:
        candidate = OUTPUT_DIR / f"{image_path.stem}_hybrid_{index}.jpg"
        if not candidate.exists():
            return candidate
        index += 1


def clamp_box(
    box_xyxy: list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box_xyxy
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def classify_roi(
    roi: Image.Image,
    model: nn.Module,
    device: torch.device,
) -> tuple[str, float]:
    tensor = EFFICIENTNET_TRANSFORM(roi).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, class_index = torch.max(probabilities, dim=0)

    class_id = int(class_index.item())
    return CLASS_NAMES[class_id], float(confidence.item())


def draw_label(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, _y2 = box
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((x1, y1), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_top = max(0, y1 - text_height - 6)
    draw.rectangle(
        [x1, text_top, min(x1 + text_width + 6, x2), text_top + text_height + 6],
        fill=color,
    )
    draw.text((x1 + 3, text_top + 3), text, fill=(255, 255, 255), font=font)


def draw_no_detection_status(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = "No gingivitis detected"
    text_bbox = draw.textbbox((10, 10), text, font=font)
    draw.rectangle(
        [8, 8, text_bbox[2] + 14, text_bbox[3] + 14],
        fill=(30, 30, 30),
    )
    draw.text((11, 11), text, fill=(255, 255, 255), font=font)


def run_inference(image_path: Path) -> Path:
    image_path = validate_file(image_path, "Input image")
    yolo_path = validate_file(YOLO_MODEL_PATH, "YOLO model")
    efficientnet_path = validate_file(EFFICIENTNET_MODEL_PATH, "EfficientNet model")

    image = load_image(image_path)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = image.size

    device = select_device()

    try:
        yolo_model = load_yolo(yolo_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model: {yolo_path}") from exc

    try:
        severity_model = load_efficientnet(efficientnet_path, device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load EfficientNet model: {efficientnet_path}. {exc}"
        ) from exc

    try:
        results = yolo_model.predict(
            source=str(image_path),
            conf=YOLO_CONFIDENCE_THRESHOLD,
            verbose=False,
            device=0 if device.type == "cuda" else "cpu",
        )
    except Exception as exc:
        raise RuntimeError(f"YOLO inference failed for image: {image_path}") from exc

    detections: list[dict[str, Any]] = []
    skipped_zero_area = 0

    if results:
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                yolo_confidence = float(box.conf.detach().cpu().item())
                raw_box = box.xyxy.detach().cpu().numpy()[0].tolist()
                safe_box = clamp_box(raw_box, width=width, height=height)
                if safe_box is None:
                    skipped_zero_area += 1
                    continue

                roi = image.crop(safe_box)
                severity, severity_confidence = classify_roi(roi, severity_model, device)

                detections.append(
                    {
                        "box": safe_box,
                        "yolo_confidence": yolo_confidence,
                        "severity": severity,
                        "severity_confidence": severity_confidence,
                    }
                )

    print(f"Image: {image_path.name}")
    print()

    if not detections:
        print("No gingivitis detected")
        if skipped_zero_area:
            print(f"Skipped zero-area crops: {skipped_zero_area}")
        draw_no_detection_status(annotated)
    else:
        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection["box"]
            label = (
                f"Gingivitis {detection['yolo_confidence']:.2f} | "
                f"{detection['severity']} {detection['severity_confidence']:.2f}"
            )
            draw.rectangle([x1, y1, x2, y2], outline=(220, 40, 40), width=4)
            draw_label(draw, detection["box"], label, color=(180, 30, 30))

            print(f"Detection {index}")
            print(f"YOLO confidence: {detection['yolo_confidence']:.2f}")
            print(f"Severity: {detection['severity']}")
            print(f"Severity confidence: {detection['severity_confidence']:.2f}")
            print(f"Box: [{x1}, {y1}, {x2}, {y2}]")
            print()

        if skipped_zero_area:
            print(f"Skipped zero-area crops: {skipped_zero_area}")

    output_path = safe_output_path(image_path)
    annotated.save(output_path, quality=95)
    print(f"Output saved: {output_path}")
    return output_path


def main() -> int:
    args = parse_args()
    try:
        run_inference(args.image)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
