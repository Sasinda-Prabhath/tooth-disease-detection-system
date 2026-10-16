"""Extract four-class gingivitis classification ROI crops.

The script reads original image/label pairs, crops each valid gingival YOLO
box, and writes crops into generated split/class folders. Original files are
read only.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - user-facing dependency guard.
    raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc


SPLITS = ("Training", "Validation", "Test")
CLASS_NAMES = ("Healthy", "Mild", "Moderate", "Severe")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SOURCE_CLASS_NAMES = {
    0: "Maxilla",
    1: "Mandible",
    2: "MGI0",
    3: "MGI1",
    4: "MGI2",
    5: "MGI3",
    6: "MGI4",
}

SOURCE_TO_CLASS_NAME = {
    2: "Healthy",
    3: "Mild",
    4: "Mild",
    5: "Moderate",
    6: "Severe",
}

MIN_ROI_SIZE_PX = 2


@dataclass
class RoiStats:
    total_images: int = 0
    total_annotations: int = 0
    generated_rois: int = 0
    class_counts: Counter[str] = field(default_factory=Counter)
    removed_maxilla: int = 0
    removed_mandible: int = 0
    invalid_label_lines: int = 0
    skipped_rois: int = 0
    missing_label_files: int = 0
    missing_images: int = 0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_root(source_root: Path) -> Path:
    return source_root / "datasets" / "gingivitis"


def find_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def find_labels(labels_dir: Path) -> list[Path]:
    if not labels_dir.exists():
        return []
    return sorted(labels_dir.glob("*.txt"), key=lambda path: path.name.lower())


def ensure_classification_structure(generated_root: Path) -> None:
    (generated_root / "reports").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (generated_root / "classification_levels" / split / class_name).mkdir(parents=True, exist_ok=True)


def parse_yolo_line(raw_line: str) -> tuple[int, tuple[float, float, float, float] | None, str | None]:
    parts = raw_line.split()
    if len(parts) != 5:
        return -1, None, "expected 5 YOLO columns"

    try:
        source_class = int(parts[0])
    except ValueError:
        return -1, None, "class id is not an integer"

    if source_class not in SOURCE_CLASS_NAMES:
        return -1, None, f"unknown source class {parts[0]}"

    try:
        x_center, y_center, width, height = [float(value) for value in parts[1:]]
    except ValueError:
        return -1, None, "one or more coordinates are not numeric"

    values = (x_center, y_center, width, height)
    if not all(math.isfinite(value) for value in values):
        return -1, None, "one or more coordinates are not finite"
    if width <= 0 or height <= 0:
        return -1, None, "width and height must be positive"
    return source_class, values, None


def yolo_to_pixel_box(
    coords: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x_center, y_center, width, height = coords
    box_width = width * image_width
    box_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height

    left = math.floor(center_x - box_width / 2)
    top = math.floor(center_y - box_height / 2)
    right = math.ceil(center_x + box_width / 2)
    bottom = math.ceil(center_y + box_height / 2)

    left = max(0, min(image_width, left))
    top = max(0, min(image_height, top))
    right = max(0, min(image_width, right))
    bottom = max(0, min(image_height, bottom))
    return left, top, right, bottom


def write_roi_counts(reports_root: Path, stats_by_split: dict[str, RoiStats]) -> None:
    path = reports_root / "classification_roi_counts.csv"
    fieldnames = [
        "split",
        "Healthy",
        "Mild",
        "Moderate",
        "Severe",
        "total_images",
        "total_annotations",
        "generated_roi_count",
        "removed_maxilla_annotations",
        "removed_mandible_annotations",
        "invalid_label_lines",
        "missing_label_files",
        "missing_images",
        "skipped_rois",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        totals = RoiStats()
        for split in SPLITS:
            stats = stats_by_split[split]
            writer.writerow(stats_row(split, stats))
            add_stats(totals, stats)
        writer.writerow(stats_row("Total", totals))


def stats_row(split: str, stats: RoiStats) -> dict[str, int | str]:
    return {
        "split": split,
        "Healthy": stats.class_counts["Healthy"],
        "Mild": stats.class_counts["Mild"],
        "Moderate": stats.class_counts["Moderate"],
        "Severe": stats.class_counts["Severe"],
        "total_images": stats.total_images,
        "total_annotations": stats.total_annotations,
        "generated_roi_count": stats.generated_rois,
        "removed_maxilla_annotations": stats.removed_maxilla,
        "removed_mandible_annotations": stats.removed_mandible,
        "invalid_label_lines": stats.invalid_label_lines,
        "missing_label_files": stats.missing_label_files,
        "missing_images": stats.missing_images,
        "skipped_rois": stats.skipped_rois,
    }


def add_stats(total: RoiStats, stats: RoiStats) -> None:
    total.total_images += stats.total_images
    total.total_annotations += stats.total_annotations
    total.generated_rois += stats.generated_rois
    total.class_counts.update(stats.class_counts)
    total.removed_maxilla += stats.removed_maxilla
    total.removed_mandible += stats.removed_mandible
    total.invalid_label_lines += stats.invalid_label_lines
    total.skipped_rois += stats.skipped_rois
    total.missing_label_files += stats.missing_label_files
    total.missing_images += stats.missing_images


def extract_rois(source_root: Path, generated_root: Path) -> dict[str, RoiStats]:
    classification_root = generated_root / "classification_levels"
    reports_root = generated_root / "reports"
    skipped_path = reports_root / "skipped_rois.txt"
    stats_by_split: dict[str, RoiStats] = {}

    with skipped_path.open("w", encoding="utf-8") as skipped:
        skipped.write("Skipped classification ROIs\n")
        skipped.write("=" * 28 + "\n")

        for split in SPLITS:
            images_dir = source_root / split / "Images"
            labels_dir = source_root / split / "Labels"
            images = find_images(images_dir)
            labels = find_labels(labels_dir)
            image_by_stem = {path.stem: path for path in images}
            label_by_stem = {path.stem: path for path in labels}

            stats = RoiStats(total_images=len(images))
            stats.missing_label_files = sum(1 for image in images if image.stem not in label_by_stem)
            stats.missing_images = sum(1 for label in labels if label.stem not in image_by_stem)
            stats_by_split[split] = stats

            print(f"[roi] {split}: extracting ROIs from {len(images)} images")

            for image_path in images:
                label_path = label_by_stem.get(image_path.stem)
                if label_path is None:
                    skipped.write(f"{split}: missing label for image {image_path.name}\n")
                    continue

                try:
                    with Image.open(image_path) as image:
                        image = image.convert("RGB")
                        image_width, image_height = image.size
                        roi_number = 0

                        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                            line = raw_line.strip()
                            if not line:
                                continue

                            source_class, coords, error = parse_yolo_line(line)
                            if error:
                                stats.invalid_label_lines += 1
                                skipped.write(f"{split}: {label_path.name}:{line_number}: {error}: {raw_line}\n")
                                continue

                            stats.total_annotations += 1
                            if source_class == 0:
                                stats.removed_maxilla += 1
                                continue
                            if source_class == 1:
                                stats.removed_mandible += 1
                                continue

                            class_name = SOURCE_TO_CLASS_NAME[source_class]
                            assert coords is not None
                            left, top, right, bottom = yolo_to_pixel_box(coords, image_width, image_height)

                            if right <= left or bottom <= top:
                                stats.skipped_rois += 1
                                skipped.write(f"{split}: {label_path.name}:{line_number}: ROI outside image bounds\n")
                                continue
                            if (right - left) < MIN_ROI_SIZE_PX or (bottom - top) < MIN_ROI_SIZE_PX:
                                stats.skipped_rois += 1
                                skipped.write(
                                    f"{split}: {label_path.name}:{line_number}: ROI too small "
                                    f"({right - left}x{bottom - top})\n"
                                )
                                continue

                            crop = image.crop((left, top, right, bottom))
                            if crop.size[0] == 0 or crop.size[1] == 0:
                                stats.skipped_rois += 1
                                skipped.write(f"{split}: {label_path.name}:{line_number}: empty crop\n")
                                continue

                            roi_number += 1
                            file_name = f"{image_path.stem}_{class_name}_roi_{roi_number:03d}.jpg"
                            out_path = classification_root / split / class_name / file_name
                            crop.save(out_path, format="JPEG", quality=95)
                            stats.generated_rois += 1
                            stats.class_counts[class_name] += 1
                except OSError as exc:
                    stats.skipped_rois += 1
                    skipped.write(f"{split}: {image_path.name}: could not open image: {exc}\n")

            for label in labels:
                if label.stem not in image_by_stem:
                    skipped.write(f"{split}: label has no matching image {label.name}\n")

    write_roi_counts(reports_root, stats_by_split)
    return stats_by_split


def print_counts(stats_by_split: dict[str, RoiStats]) -> None:
    print("")
    print("Classification ROI counts")
    print("=========================")
    totals = RoiStats()
    for split in SPLITS:
        stats = stats_by_split[split]
        add_stats(totals, stats)
        print(
            f"{split}: Healthy={stats.class_counts['Healthy']}, Mild={stats.class_counts['Mild']}, "
            f"Moderate={stats.class_counts['Moderate']}, Severe={stats.class_counts['Severe']}, "
            f"generated_rois={stats.generated_rois}"
        )
    print(
        f"Total: Healthy={totals.class_counts['Healthy']}, Mild={totals.class_counts['Mild']}, "
        f"Moderate={totals.class_counts['Moderate']}, Severe={totals.class_counts['Severe']}, "
        f"generated_rois={totals.generated_rois}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract gingivitis severity classification ROI crops.")
    parser.add_argument("--source-root", type=Path, default=project_root(), help="Original dataset root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Generated dataset output root.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    source_root = args.source_root.resolve()
    generated_root = args.output_root.resolve() if args.output_root else output_root(source_root)

    ensure_classification_structure(generated_root)
    stats_by_split = extract_rois(source_root, generated_root)
    print_counts(stats_by_split)
    print("")
    print("Original files were not edited, moved, renamed, overwritten, or deleted.")
    print("Classification crops preserve original split membership.")


if __name__ == "__main__":
    main()
