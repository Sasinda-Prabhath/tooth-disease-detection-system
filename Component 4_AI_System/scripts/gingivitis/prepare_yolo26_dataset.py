"""Prepare a four-class YOLO gingivitis detection dataset.

This script reads the original read-only Training/Validation/Test folders,
copies images into a generated dataset folder, remaps labels, and writes audit
reports. It never edits, moves, or deletes original files.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


SPLITS = ("Training", "Validation", "Test")
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

DETECTION_CLASS_NAMES = {
    0: "Healthy",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
}

SOURCE_TO_DETECTION = {
    2: 0,
    3: 1,
    4: 1,
    5: 2,
    6: 3,
}


@dataclass
class SplitAudit:
    split: str
    images_dir: Path
    labels_dir: Path
    images: list[Path] = field(default_factory=list)
    labels: list[Path] = field(default_factory=list)
    missing_label_files: list[Path] = field(default_factory=list)
    labels_without_images: list[Path] = field(default_factory=list)
    image_extensions: Counter[str] = field(default_factory=Counter)


@dataclass
class DetectionStats:
    total_images: int = 0
    total_label_files: int = 0
    total_annotations: int = 0
    class_counts: Counter[str] = field(default_factory=Counter)
    removed_maxilla: int = 0
    removed_mandible: int = 0
    invalid_label_lines: int = 0
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


def audit_dataset(source_root: Path) -> dict[str, SplitAudit]:
    audits: dict[str, SplitAudit] = {}
    for split in SPLITS:
        split_root = source_root / split
        images_dir = split_root / "Images"
        labels_dir = split_root / "Labels"
        images = find_images(images_dir)
        labels = find_labels(labels_dir)
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}

        audit = SplitAudit(
            split=split,
            images_dir=images_dir,
            labels_dir=labels_dir,
            images=images,
            labels=labels,
            missing_label_files=[path for path in images if path.stem not in label_stems],
            labels_without_images=[path for path in labels if path.stem not in image_stems],
            image_extensions=Counter(path.suffix.lower() for path in images),
        )
        audits[split] = audit
    return audits


def parse_yolo_line(label_path: Path, line_number: int, line: str) -> tuple[int, list[str], str | None]:
    parts = line.split()
    if len(parts) != 5:
        return -1, [], "expected 5 YOLO columns"

    try:
        source_class = int(parts[0])
    except ValueError:
        return -1, [], "class id is not an integer"

    if source_class not in SOURCE_CLASS_NAMES:
        return -1, [], f"unknown source class {parts[0]}"

    try:
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return -1, [], "one or more coordinates are not numeric"

    if not all(math.isfinite(value) for value in coords):
        return -1, [], "one or more coordinates are not finite"

    if coords[2] <= 0 or coords[3] <= 0:
        return -1, [], "width and height must be positive"

    return source_class, parts[1:], None


def csv_and_yaml_files(source_root: Path) -> tuple[list[Path], list[Path]]:
    files = [path for path in source_root.rglob("*") if path.is_file()]
    csv_files = sorted([path for path in files if path.suffix.lower() == ".csv"])
    yaml_files = sorted([path for path in files if path.suffix.lower() in {".yaml", ".yml"}])
    return csv_files, yaml_files


def ensure_output_structure(base: Path) -> None:
    (base / "original_1096").mkdir(parents=True, exist_ok=True)
    (base / "reports").mkdir(parents=True, exist_ok=True)
    (base / "synthetic").mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (base / "detection_yolo26" / split / "images").mkdir(parents=True, exist_ok=True)
        (base / "detection_yolo26" / split / "labels").mkdir(parents=True, exist_ok=True)


def write_data_yaml(detection_root: Path) -> None:
    yaml_path = detection_root / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: {detection_root.as_posix()}",
            "train: Training/images",
            "val: Validation/images",
            "test: Test/images",
            "nc: 4",
            "names:",
            "  0: Healthy",
            "  1: Mild",
            "  2: Moderate",
            "  3: Severe",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")


def write_audit_report(
    reports_root: Path,
    source_root: Path,
    generated_root: Path,
    audits: dict[str, SplitAudit],
    stats_by_split: dict[str, DetectionStats],
) -> None:
    csv_files, yaml_files = csv_and_yaml_files(source_root)
    lines: list[str] = []
    lines.append("Gingivitis Dataset Audit")
    lines.append("=" * 26)
    lines.append(f"Source dataset path: {source_root}")
    lines.append(f"Generated output path: {generated_root}")
    lines.append("")
    lines.append("Detected split folders:")

    for split in SPLITS:
        audit = audits[split]
        stats = stats_by_split.get(split, DetectionStats())
        ext_summary = ", ".join(f"{ext}={count}" for ext, count in sorted(audit.image_extensions.items()))
        lines.extend(
            [
                f"- {split}",
                f"  Images folder: {audit.images_dir} (exists={audit.images_dir.exists()})",
                f"  Labels folder: {audit.labels_dir} (exists={audit.labels_dir.exists()})",
                f"  Image files: {len(audit.images)}",
                f"  Label files: {len(audit.labels)}",
                f"  Image extensions: {ext_summary or 'none'}",
                f"  Images missing labels: {len(audit.missing_label_files)}",
                f"  Labels missing images: {len(audit.labels_without_images)}",
                f"  Total source annotations: {stats.total_annotations}",
                f"  Removed Maxilla annotations: {stats.removed_maxilla}",
                f"  Removed Mandible annotations: {stats.removed_mandible}",
                f"  Invalid label lines: {stats.invalid_label_lines}",
            ]
        )

    total_images = sum(len(audits[split].images) for split in SPLITS)
    total_labels = sum(len(audits[split].labels) for split in SPLITS)
    total_annotations = sum(stats.total_annotations for stats in stats_by_split.values())
    total_invalid = sum(stats.invalid_label_lines for stats in stats_by_split.values())
    total_maxilla = sum(stats.removed_maxilla for stats in stats_by_split.values())
    total_mandible = sum(stats.removed_mandible for stats in stats_by_split.values())
    lines.extend(
        [
            "",
            "Totals:",
            f"  Total images: {total_images}",
            f"  Total label files: {total_labels}",
            f"  Total annotations: {total_annotations}",
            f"  Removed Maxilla annotations: {total_maxilla}",
            f"  Removed Mandible annotations: {total_mandible}",
            f"  Invalid label lines: {total_invalid}",
            "",
            "CSV files:",
        ]
    )
    lines.extend(f"  {path}" for path in csv_files)
    lines.append("YAML files:")
    if yaml_files:
        lines.extend(f"  {path}" for path in yaml_files)
    else:
        lines.append("  none found")

    (reports_root / "dataset_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_detection_counts(reports_root: Path, stats_by_split: dict[str, DetectionStats]) -> None:
    path = reports_root / "detection_class_counts.csv"
    fieldnames = [
        "split",
        "Healthy",
        "Mild",
        "Moderate",
        "Severe",
        "total_images",
        "total_annotations",
        "removed_maxilla_annotations",
        "removed_mandible_annotations",
        "invalid_label_lines",
        "missing_label_files",
        "missing_images",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        totals = DetectionStats()
        for split in SPLITS:
            stats = stats_by_split[split]
            writer.writerow(stats_row(split, stats))
            add_stats(totals, stats)
        writer.writerow(stats_row("Total", totals))


def stats_row(split: str, stats: DetectionStats) -> dict[str, int | str]:
    return {
        "split": split,
        "Healthy": stats.class_counts["Healthy"],
        "Mild": stats.class_counts["Mild"],
        "Moderate": stats.class_counts["Moderate"],
        "Severe": stats.class_counts["Severe"],
        "total_images": stats.total_images,
        "total_annotations": stats.total_annotations,
        "removed_maxilla_annotations": stats.removed_maxilla,
        "removed_mandible_annotations": stats.removed_mandible,
        "invalid_label_lines": stats.invalid_label_lines,
        "missing_label_files": stats.missing_label_files,
        "missing_images": stats.missing_images,
    }


def add_stats(total: DetectionStats, stats: DetectionStats) -> None:
    total.total_images += stats.total_images
    total.total_label_files += stats.total_label_files
    total.total_annotations += stats.total_annotations
    total.class_counts.update(stats.class_counts)
    total.removed_maxilla += stats.removed_maxilla
    total.removed_mandible += stats.removed_mandible
    total.invalid_label_lines += stats.invalid_label_lines
    total.missing_label_files += stats.missing_label_files
    total.missing_images += stats.missing_images


def prepare_detection_dataset(source_root: Path, generated_root: Path, audits: dict[str, SplitAudit]) -> dict[str, DetectionStats]:
    detection_root = generated_root / "detection_yolo26"
    reports_root = generated_root / "reports"
    skipped_log = reports_root / "skipped_annotations.txt"
    stats_by_split: dict[str, DetectionStats] = {}

    with skipped_log.open("w", encoding="utf-8") as skipped:
        skipped.write("Skipped/removed detection annotations\n")
        skipped.write("=" * 38 + "\n")

        for split in SPLITS:
            audit = audits[split]
            stats = DetectionStats(
                total_images=len(audit.images),
                total_label_files=len(audit.labels),
                missing_label_files=len(audit.missing_label_files),
                missing_images=len(audit.labels_without_images),
            )
            stats_by_split[split] = stats

            out_images_dir = detection_root / split / "images"
            out_labels_dir = detection_root / split / "labels"
            print(f"[prepare] {split}: copying {len(audit.images)} images and remapping labels")

            for image_path in audit.images:
                shutil.copy2(image_path, out_images_dir / image_path.name)

            for missing in audit.missing_label_files:
                skipped.write(f"{split}: missing label file for image {missing.name}\n")
            for missing in audit.labels_without_images:
                skipped.write(f"{split}: label has no matching image {missing.name}\n")

            for label_path in audit.labels:
                output_lines: list[str] = []
                for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                    line = raw_line.strip()
                    if not line:
                        continue

                    source_class, coord_tokens, error = parse_yolo_line(label_path, line_number, line)
                    if error:
                        stats.invalid_label_lines += 1
                        skipped.write(f"{split}: {label_path.name}:{line_number}: {error}: {raw_line}\n")
                        continue

                    stats.total_annotations += 1
                    if source_class == 0:
                        stats.removed_maxilla += 1
                        skipped.write(f"{split}: {label_path.name}:{line_number}: removed Maxilla annotation\n")
                        continue
                    if source_class == 1:
                        stats.removed_mandible += 1
                        skipped.write(f"{split}: {label_path.name}:{line_number}: removed Mandible annotation\n")
                        continue

                    detection_class = SOURCE_TO_DETECTION[source_class]
                    class_name = DETECTION_CLASS_NAMES[detection_class]
                    stats.class_counts[class_name] += 1
                    output_lines.append(f"{detection_class} {' '.join(coord_tokens)}")

                (out_labels_dir / label_path.name).write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

    write_data_yaml(detection_root)
    write_detection_counts(reports_root, stats_by_split)
    write_audit_report(reports_root, source_root, generated_root, audits, stats_by_split)
    return stats_by_split


def print_audit(audits: dict[str, SplitAudit], source_root: Path, generated_root: Path) -> None:
    csv_files, yaml_files = csv_and_yaml_files(source_root)
    print("Dataset audit before processing")
    print("==============================")
    print(f"Source dataset path: {source_root}")
    print(f"Proposed output path: {generated_root}")
    print("Files/scripts created by this workflow:")
    print("  scripts/gingivitis/prepare_yolo26_dataset.py")
    print("  scripts/gingivitis/extract_classification_rois.py")
    print("  scripts/gingivitis/analyze_dataset_balance.py")
    print("  datasets/gingivitis/detection_yolo26/")
    print("  datasets/gingivitis/classification_levels/")
    print("  datasets/gingivitis/reports/")
    print("  datasets/gingivitis/original_1096/")
    print("  datasets/gingivitis/synthetic/")
    print("")
    for split in SPLITS:
        audit = audits[split]
        ext_summary = ", ".join(f"{ext}={count}" for ext, count in sorted(audit.image_extensions.items()))
        print(f"{split}:")
        print(f"  images folder: {audit.images_dir} exists={audit.images_dir.exists()}")
        print(f"  labels folder: {audit.labels_dir} exists={audit.labels_dir.exists()}")
        print(f"  images: {len(audit.images)}")
        print(f"  labels: {len(audit.labels)}")
        print(f"  image extensions: {ext_summary or 'none'}")
        print(f"  images missing labels: {len(audit.missing_label_files)}")
        print(f"  labels missing images: {len(audit.labels_without_images)}")
    print("CSV files:")
    for path in csv_files:
        print(f"  {path}")
    print("YAML files:")
    if yaml_files:
        for path in yaml_files:
            print(f"  {path}")
    else:
        print("  none found")


def print_counts(stats_by_split: dict[str, DetectionStats]) -> None:
    print("")
    print("Detection class counts")
    print("======================")
    totals = DetectionStats()
    for split in SPLITS:
        stats = stats_by_split[split]
        add_stats(totals, stats)
        print(
            f"{split}: Healthy={stats.class_counts['Healthy']}, Mild={stats.class_counts['Mild']}, "
            f"Moderate={stats.class_counts['Moderate']}, Severe={stats.class_counts['Severe']}"
        )
    print(
        f"Total: Healthy={totals.class_counts['Healthy']}, Mild={totals.class_counts['Mild']}, "
        f"Moderate={totals.class_counts['Moderate']}, Severe={totals.class_counts['Severe']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a four-class YOLO26 gingivitis detection dataset.")
    parser.add_argument("--source-root", type=Path, default=project_root(), help="Original dataset root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Generated dataset output root.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    source_root = args.source_root.resolve()
    generated_root = args.output_root.resolve() if args.output_root else output_root(source_root)

    audits = audit_dataset(source_root)
    print_audit(audits, source_root, generated_root)
    ensure_output_structure(generated_root)
    stats_by_split = prepare_detection_dataset(source_root, generated_root, audits)
    print_counts(stats_by_split)
    print("")
    print("Original files were not edited, moved, renamed, overwritten, or deleted.")
    print("Training/Validation/Test split membership was preserved.")


if __name__ == "__main__":
    main()
