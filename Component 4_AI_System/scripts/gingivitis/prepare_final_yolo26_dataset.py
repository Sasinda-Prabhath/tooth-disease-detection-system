from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_SOURCE = Path(r"D:\Gingivities Dataset\detection_yolo26")
DEFAULT_TARGET = Path(r"D:\Gingivitis_YOLO26_Final")
SPLITS = ("Training", "Validation", "Test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ConvertedLabelStats:
    gingivitis_boxes: int = 0
    removed_healthy: int = 0
    converted_mild: int = 0
    converted_moderate: int = 0
    converted_severe: int = 0
    source_other_classes: int = 0
    invalid_source_lines: int = 0
    source_label_exists: bool = False


@dataclass
class SplitAudit:
    split: str
    image_count: int = 0
    label_file_count: int = 0
    gingivitis_boxes: int = 0
    images_with_gingivitis: int = 0
    negative_images: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    invalid_yolo: list[str] = field(default_factory=list)
    unreadable_images: list[str] = field(default_factory=list)
    class_ids: Counter = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and audit the final one-class YOLO26 gingivitis dataset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    return parser.parse_args()


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def require_safe_paths(source: Path, target: Path) -> tuple[Path, Path]:
    source = source.resolve()
    target = target.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source dataset is not a directory: {source}")
    if target.exists():
        raise FileExistsError(
            f"Target already exists and will not be overwritten: {target}"
        )
    if source == target or is_relative_to(target, source):
        raise ValueError("Target must not be the source directory or inside it.")

    for split in SPLITS:
        for child in ("images", "labels"):
            required = source / split / child
            if not required.is_dir():
                raise FileNotFoundError(f"Required source directory missing: {required}")

    return source, target


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )


def iter_label_files(label_dir: Path) -> list[Path]:
    return sorted(
        (path for path in label_dir.rglob("*.txt") if path.is_file()),
        key=lambda path: path.as_posix().lower(),
    )


def image_to_label_relpath(image_relpath: Path) -> Path:
    return image_relpath.with_suffix(".txt")


def read_text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def convert_label(source_label: Path) -> tuple[list[str], ConvertedLabelStats]:
    stats = ConvertedLabelStats(source_label_exists=source_label.exists())
    converted_lines: list[str] = []

    if not source_label.exists():
        return converted_lines, stats

    for raw_line in read_text_lines(source_label):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            stats.invalid_source_lines += 1
            continue

        old_class = parts[0]
        coords = parts[1:5]

        if old_class == "0":
            stats.removed_healthy += 1
            continue
        if old_class == "1":
            stats.converted_mild += 1
            converted_lines.append("0 " + " ".join(coords))
            continue
        if old_class == "2":
            stats.converted_moderate += 1
            converted_lines.append("0 " + " ".join(coords))
            continue
        if old_class == "3":
            stats.converted_severe += 1
            converted_lines.append("0 " + " ".join(coords))
            continue

        stats.source_other_classes += 1
        stats.invalid_source_lines += 1

    stats.gingivitis_boxes = len(converted_lines)
    return converted_lines, stats


def write_label(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def copy_and_convert(source: Path, target: Path) -> dict[str, dict[str, ConvertedLabelStats]]:
    target.mkdir(parents=True)
    (target / "reports").mkdir()

    data_yaml = "\n".join(
        [
            f"path: {target.as_posix()}",
            "train: Training/images",
            "val: Validation/images",
            "test: Test/images",
            "nc: 1",
            "names:",
            "  0: Gingivitis",
            "",
        ]
    )
    (target / "data.yaml").write_text(data_yaml, encoding="utf-8", newline="\n")

    conversion: dict[str, dict[str, ConvertedLabelStats]] = {
        split: {} for split in SPLITS
    }

    for split in SPLITS:
        source_images_dir = source / split / "images"
        source_labels_dir = source / split / "labels"
        target_images_dir = target / split / "images"
        target_labels_dir = target / split / "labels"
        target_images_dir.mkdir(parents=True)
        target_labels_dir.mkdir(parents=True)

        for image_path in iter_images(source_images_dir):
            rel_image = image_path.relative_to(source_images_dir)
            rel_label = image_to_label_relpath(rel_image)
            target_image = target_images_dir / rel_image
            target_label = target_labels_dir / rel_label
            source_label = source_labels_dir / rel_label

            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)

            converted_lines, stats = convert_label(source_label)
            write_label(target_label, converted_lines)
            conversion[split][rel_image.as_posix()] = stats

    return conversion


def parse_final_label(label_path: Path, split: str, rel_label: str) -> tuple[int, list[str], Counter]:
    boxes = 0
    invalid: list[str] = []
    class_ids: Counter = Counter()

    for line_number, raw_line in enumerate(read_text_lines(label_path), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()

        if len(parts) != 5:
            invalid.append(
                f"{split}/{rel_label}:{line_number}: expected 5 YOLO fields, found {len(parts)}"
            )
            continue

        class_id = parts[0]
        class_ids[class_id] += 1
        if class_id != "0":
            invalid.append(
                f"{split}/{rel_label}:{line_number}: class id is {class_id}, expected 0"
            )

        try:
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            invalid.append(
                f"{split}/{rel_label}:{line_number}: non-numeric YOLO coordinate"
            )
            continue

        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            invalid.append(
                f"{split}/{rel_label}:{line_number}: center coordinate outside [0, 1]"
            )
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            invalid.append(
                f"{split}/{rel_label}:{line_number}: width/height outside (0, 1]"
            )

        boxes += 1

    return boxes, invalid, class_ids


def verify_image(image_path: Path, split: str, rel_image: str) -> str | None:
    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - report exact image verification failure
        return f"{split}/{rel_image}: {exc}"
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(target: Path) -> tuple[dict[str, SplitAudit], dict[str, list[tuple[str, str]]], dict[str, set[str]]]:
    audits: dict[str, SplitAudit] = {}
    hash_to_images: dict[str, list[tuple[str, str]]] = defaultdict(list)
    rel_names_by_split: dict[str, set[str]] = {}

    for split in SPLITS:
        images_dir = target / split / "images"
        labels_dir = target / split / "labels"
        image_paths = iter_images(images_dir)
        label_paths = iter_label_files(labels_dir)

        audit = SplitAudit(
            split=split,
            image_count=len(image_paths),
            label_file_count=len(label_paths),
        )

        image_relpaths = {
            image.relative_to(images_dir).as_posix(): image for image in image_paths
        }
        label_relpaths = {
            label.relative_to(labels_dir).as_posix(): label for label in label_paths
        }
        expected_label_relpaths = {
            image_to_label_relpath(Path(rel_image)).as_posix()
            for rel_image in image_relpaths
        }

        audit.missing_labels = sorted(expected_label_relpaths - set(label_relpaths))
        audit.orphan_labels = sorted(set(label_relpaths) - expected_label_relpaths)

        rel_names_by_split[split] = set(image_relpaths)

        boxes_by_expected_label: dict[str, int] = {}
        for rel_label in sorted(expected_label_relpaths & set(label_relpaths)):
            boxes, invalid, class_ids = parse_final_label(label_relpaths[rel_label], split, rel_label)
            boxes_by_expected_label[rel_label] = boxes
            audit.gingivitis_boxes += boxes
            audit.invalid_yolo.extend(invalid)
            audit.class_ids.update(class_ids)

        for rel_image, image_path in image_relpaths.items():
            rel_label = image_to_label_relpath(Path(rel_image)).as_posix()
            box_count = boxes_by_expected_label.get(rel_label, 0)
            if box_count > 0:
                audit.images_with_gingivitis += 1
            elif rel_label in label_relpaths:
                audit.negative_images += 1

            unreadable = verify_image(image_path, split, rel_image)
            if unreadable:
                audit.unreadable_images.append(unreadable)

            hash_to_images[sha256_file(image_path)].append((split, rel_image))

        audits[split] = audit

    overlapping_hashes = {
        image_hash: locations
        for image_hash, locations in hash_to_images.items()
        if len({split for split, _ in locations}) > 1
    }

    return audits, overlapping_hashes, rel_names_by_split


def source_orphan_labels(source: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in SPLITS:
        images_dir = source / split / "images"
        labels_dir = source / split / "labels"
        expected = {
            image_to_label_relpath(image.relative_to(images_dir)).as_posix()
            for image in iter_images(images_dir)
        }
        labels = {label.relative_to(labels_dir).as_posix() for label in iter_label_files(labels_dir)}
        result[split] = sorted(labels - expected)
    return result


def write_csv_reports(
    target: Path,
    conversion: dict[str, dict[str, ConvertedLabelStats]],
) -> None:
    reports_dir = target / "reports"

    box_counts_path = reports_dir / "final_yolo26_box_counts.csv"
    with box_counts_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "split",
                "image",
                "label_file",
                "gingivitis_boxes",
                "is_negative",
                "source_label_exists",
                "old_healthy_removed",
                "old_mild_converted",
                "old_moderate_converted",
                "old_severe_converted",
                "source_other_classes",
                "invalid_source_lines",
            ]
        )
        for split in SPLITS:
            for rel_image, stats in sorted(conversion[split].items()):
                rel_label = image_to_label_relpath(Path(rel_image)).as_posix()
                writer.writerow(
                    [
                        split,
                        rel_image,
                        rel_label,
                        stats.gingivitis_boxes,
                        int(stats.gingivitis_boxes == 0),
                        int(stats.source_label_exists),
                        stats.removed_healthy,
                        stats.converted_mild,
                        stats.converted_moderate,
                        stats.converted_severe,
                        stats.source_other_classes,
                        stats.invalid_source_lines,
                    ]
                )

    negative_path = reports_dir / "final_yolo26_negative_images.csv"
    with negative_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["split", "image", "label_file", "reason"])
        for split in SPLITS:
            for rel_image, stats in sorted(conversion[split].items()):
                if stats.gingivitis_boxes != 0:
                    continue
                rel_label = image_to_label_relpath(Path(rel_image)).as_posix()
                if not stats.source_label_exists:
                    reason = "missing_source_label_empty_final_label"
                elif stats.removed_healthy > 0:
                    reason = "healthy_only_or_no_remaining_gingivitis_boxes"
                else:
                    reason = "empty_source_label"
                writer.writerow([split, rel_image, rel_label, reason])


def format_list_preview(items: list[str], max_items: int = 25) -> list[str]:
    if len(items) <= max_items:
        return items
    return items[:max_items] + [f"... {len(items) - max_items} more"]


def build_summary_text(
    source: Path,
    target: Path,
    audits: dict[str, SplitAudit],
    overlapping_hashes: dict[str, list[tuple[str, str]]],
    rel_names_by_split: dict[str, set[str]],
    source_orphans: dict[str, list[str]],
) -> str:
    lines: list[str] = []
    lines.append("FINAL YOLO26 GINGIVITIS DATASET AUDIT")
    lines.append(f"Source: {source}")
    lines.append(f"Target: {target}")
    lines.append("")
    lines.append("Task: one-class YOLO detection")
    lines.append("Class map: 0 = Gingivitis")
    lines.append("")

    lines.append("Per-split counts")
    lines.append(
        "split,image_count,label_file_count,total_gingivitis_boxes,"
        "images_with_gingivitis,negative_images_empty_labels,missing_labels,"
        "orphan_labels,invalid_yolo_coordinates,unreadable_images,remaining_class_ids"
    )
    for split in SPLITS:
        audit = audits[split]
        class_ids = ",".join(f"{class_id}:{count}" for class_id, count in sorted(audit.class_ids.items()))
        if not class_ids:
            class_ids = "none"
        lines.append(
            f"{split},{audit.image_count},{audit.label_file_count},"
            f"{audit.gingivitis_boxes},{audit.images_with_gingivitis},"
            f"{audit.negative_images},{len(audit.missing_labels)},"
            f"{len(audit.orphan_labels)},{len(audit.invalid_yolo)},"
            f"{len(audit.unreadable_images)},{class_ids}"
        )

    lines.append("")
    all_class_ids = Counter()
    for audit in audits.values():
        all_class_ids.update(audit.class_ids)
    non_zero_classes = sorted(class_id for class_id in all_class_ids if class_id != "0")
    lines.append(f"Verify no class IDs other than 0 remain: {'PASS' if not non_zero_classes else 'FAIL'}")
    lines.append(f"Remaining class IDs: {dict(sorted(all_class_ids.items())) if all_class_ids else {}}")

    lines.append("")
    lines.append("Final dataset issues")
    for split in SPLITS:
        audit = audits[split]
        lines.append(f"{split}:")
        lines.append(f"  missing labels: {len(audit.missing_labels)}")
        for item in format_list_preview(audit.missing_labels):
            lines.append(f"    {item}")
        lines.append(f"  orphan labels: {len(audit.orphan_labels)}")
        for item in format_list_preview(audit.orphan_labels):
            lines.append(f"    {item}")
        lines.append(f"  invalid YOLO coordinates/labels: {len(audit.invalid_yolo)}")
        for item in format_list_preview(audit.invalid_yolo):
            lines.append(f"    {item}")
        lines.append(f"  unreadable images: {len(audit.unreadable_images)}")
        for item in format_list_preview(audit.unreadable_images):
            lines.append(f"    {item}")

    lines.append("")
    lines.append("Exact split overlap")
    lines.append(f"  overlapping image SHA256 hashes across splits: {len(overlapping_hashes)}")
    for image_hash, locations in list(sorted(overlapping_hashes.items()))[:25]:
        location_text = "; ".join(f"{split}/{rel_image}" for split, rel_image in locations)
        lines.append(f"    {image_hash}: {location_text}")
    if len(overlapping_hashes) > 25:
        lines.append(f"    ... {len(overlapping_hashes) - 25} more")

    split_pairs = (("Training", "Validation"), ("Training", "Test"), ("Validation", "Test"))
    lines.append("  relative image path overlap:")
    for left, right in split_pairs:
        overlap = sorted(rel_names_by_split[left] & rel_names_by_split[right])
        lines.append(f"    {left} vs {right}: {len(overlap)}")
        for item in format_list_preview(overlap, max_items=10):
            lines.append(f"      {item}")

    lines.append("")
    lines.append("Source orphan labels not copied")
    for split in SPLITS:
        lines.append(f"  {split}: {len(source_orphans[split])}")
        for item in format_list_preview(source_orphans[split], max_items=10):
            lines.append(f"    {item}")

    lines.append("")
    lines.append("Generated files")
    lines.append(f"  {target / 'data.yaml'}")
    lines.append(f"  {target / 'reports' / 'final_yolo26_dataset_summary.txt'}")
    lines.append(f"  {target / 'reports' / 'final_yolo26_box_counts.csv'}")
    lines.append(f"  {target / 'reports' / 'final_yolo26_negative_images.csv'}")
    lines.append("")

    return "\n".join(lines)


def write_summary(
    target: Path,
    source: Path,
    audits: dict[str, SplitAudit],
    overlapping_hashes: dict[str, list[tuple[str, str]]],
    rel_names_by_split: dict[str, set[str]],
    source_orphans: dict[str, list[str]],
) -> str:
    summary = build_summary_text(
        source=source,
        target=target,
        audits=audits,
        overlapping_hashes=overlapping_hashes,
        rel_names_by_split=rel_names_by_split,
        source_orphans=source_orphans,
    )
    summary_path = target / "reports" / "final_yolo26_dataset_summary.txt"
    summary_path.write_text(summary, encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    args = parse_args()
    source, target = require_safe_paths(args.source, args.target)

    conversion = copy_and_convert(source, target)
    write_csv_reports(target, conversion)

    audits, overlapping_hashes, rel_names_by_split = audit_dataset(target)
    source_orphans = source_orphan_labels(source)
    summary = write_summary(
        target=target,
        source=source,
        audits=audits,
        overlapping_hashes=overlapping_hashes,
        rel_names_by_split=rel_names_by_split,
        source_orphans=source_orphans,
    )

    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
