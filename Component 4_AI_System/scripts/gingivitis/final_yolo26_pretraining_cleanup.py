from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


TARGET = Path(r"D:\Gingivitis_YOLO26_Final")
HEALTHY_SOURCE_CANDIDATES = [
    Path(r"D:\health to up"),
    Path(r"D:\My Drive\Research-Dataset\health to up"),
    Path(r"D:\My Drive\Research-Dataset\healthy_topup\synthetic"),
    Path(r"D:\My Drive\Research-Dataset\healthy_topup\approved"),
    Path(r"D:\My Drive\Research-Dataset\health"),
]
SPLITS = ("Training", "Validation", "Test")
SPLIT_PRIORITY = {"Training": 0, "Validation": 1, "Test": 2}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ROI_MARKERS = ("roi", "crop", "cropped", "bbox", "box", "patch")


@dataclass(frozen=True)
class ImageLocation:
    split: str
    rel_image: str
    image_path: Path
    label_path: Path

    @property
    def display(self) -> str:
        return f"{self.split}/images/{self.rel_image}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final pre-training cleanup for the YOLO26 Gingivitis dataset."
    )
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument(
        "--healthy-source",
        type=Path,
        default=None,
        help="Optional Healthy source folder override.",
    )
    parser.add_argument(
        "--proposal-only",
        action="store_true",
        help="Only inspect Healthy source images and write the allocation proposal.",
    )
    parser.add_argument(
        "--execute-unique-negatives",
        action="store_true",
        help="Copy unique Healthy negatives into the dataset and run the final audit.",
    )
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )


def iter_labels(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*.txt") if path.is_file()),
        key=lambda path: path.as_posix().lower(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_rel_to_label_path(labels_dir: Path, rel_image: str) -> Path:
    return labels_dir / Path(rel_image).with_suffix(".txt")


def ensure_target(target: Path) -> Path:
    target = target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Target dataset not found: {target}")
    for split in SPLITS:
        for child in ("images", "labels"):
            required = target / split / child
            if not required.is_dir():
                raise FileNotFoundError(f"Required dataset directory missing: {required}")
    (target / "reports").mkdir(exist_ok=True)
    return target


def collect_dataset_hashes(target: Path) -> dict[str, list[ImageLocation]]:
    hashes: dict[str, list[ImageLocation]] = defaultdict(list)
    for split in SPLITS:
        images_dir = target / split / "images"
        labels_dir = target / split / "labels"
        for image_path in iter_images(images_dir):
            rel_image = image_path.relative_to(images_dir).as_posix()
            hashes[sha256_file(image_path)].append(
                ImageLocation(
                    split=split,
                    rel_image=rel_image,
                    image_path=image_path,
                    label_path=image_rel_to_label_path(labels_dir, rel_image),
                )
            )
    return hashes


def cross_split_duplicate_groups(
    hashes: dict[str, list[ImageLocation]],
) -> dict[str, list[ImageLocation]]:
    return {
        image_hash: locations
        for image_hash, locations in hashes.items()
        if len({location.split for location in locations}) > 1
    }


def choose_retained(locations: list[ImageLocation]) -> ImageLocation:
    return sorted(
        locations,
        key=lambda location: (
            -SPLIT_PRIORITY[location.split],
            location.rel_image.lower(),
        ),
    )[0]


def remove_cross_split_duplicates(target: Path) -> tuple[Path, Path, int]:
    reports_dir = target / "reports"
    csv_path = reports_dir / "final_yolo26_cross_split_duplicate_cleanup.csv"
    txt_path = reports_dir / "final_yolo26_cross_split_duplicate_cleanup.txt"

    duplicate_groups = cross_split_duplicate_groups(collect_dataset_hashes(target))
    rows: list[dict[str, str]] = []
    text_lines = [
        "FINAL YOLO26 CROSS-SPLIT DUPLICATE CLEANUP",
        f"Target: {target}",
        f"Duplicate hash groups before cleanup: {len(duplicate_groups)}",
        "",
    ]

    for image_hash, locations in sorted(duplicate_groups.items()):
        retained = choose_retained(locations)
        removed = [location for location in locations if location != retained]
        splits = sorted({location.split for location in locations}, key=lambda split: SPLIT_PRIORITY[split])

        for location in removed:
            location.image_path.unlink()
            if location.label_path.exists():
                location.label_path.unlink()

        row = {
            "hash": image_hash,
            "duplicate_files": "; ".join(location.display for location in locations),
            "splits_involved": "; ".join(splits),
            "retained_file": retained.display,
            "removed_files": "; ".join(location.display for location in removed),
            "removed_labels": "; ".join(
                f"{location.split}/labels/{Path(location.rel_image).with_suffix('.txt').as_posix()}"
                for location in removed
            ),
        }
        rows.append(row)

        text_lines.append(f"hash: {image_hash}")
        text_lines.append(f"  splits involved: {row['splits_involved']}")
        text_lines.append(f"  duplicate files: {row['duplicate_files']}")
        text_lines.append(f"  retained file: {row['retained_file']}")
        text_lines.append(f"  removed files: {row['removed_files']}")
        text_lines.append(f"  removed labels: {row['removed_labels']}")
        text_lines.append("")

    remaining = len(cross_split_duplicate_groups(collect_dataset_hashes(target)))
    text_lines.append(f"Duplicate hash groups after cleanup: {remaining}")
    text_lines.append(f"Verification: {'PASS' if remaining == 0 else 'FAIL'}")
    text_lines.append("")

    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        fieldnames = [
            "hash",
            "duplicate_files",
            "splits_involved",
            "retained_file",
            "removed_files",
            "removed_labels",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    txt_path.write_text("\n".join(text_lines), encoding="utf-8", newline="\n")
    return csv_path, txt_path, remaining


def path_is_roi_like(path: Path) -> bool:
    text = path.as_posix().lower()
    return any(marker in text for marker in ROI_MARKERS)


def image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def inspect_source_folder(folder: Path) -> dict[str, object]:
    images = [path for path in iter_images(folder) if not path_is_roi_like(path)]
    all_images = iter_images(folder)
    hash_groups: dict[str, list[Path]] = defaultdict(list)
    unreadable: list[str] = []
    dimensions: Counter[str] = Counter()

    for image_path in images:
        dims = image_dimensions(image_path)
        if dims is None:
            unreadable.append(str(image_path))
        else:
            dimensions[f"{dims[0]}x{dims[1]}"] += 1
        hash_groups[sha256_file(image_path)].append(image_path)

    duplicate_groups = {image_hash: paths for image_hash, paths in hash_groups.items() if len(paths) > 1}
    duplicate_extra_files = sum(len(paths) - 1 for paths in duplicate_groups.values())
    return {
        "folder": folder,
        "exists": folder.exists(),
        "all_image_count": len(all_images),
        "non_roi_image_count": len(images),
        "duplicate_hash_group_count": len(duplicate_groups),
        "duplicate_extra_file_count": duplicate_extra_files,
        "unreadable_count": len(unreadable),
        "top_dimensions": dimensions.most_common(8),
        "images": images,
        "duplicate_groups": duplicate_groups,
        "unreadable": unreadable,
    }


def select_healthy_source(override: Path | None) -> dict[str, object]:
    candidates = [override] if override is not None else HEALTHY_SOURCE_CANDIDATES
    inspections = [inspect_source_folder(candidate) for candidate in candidates if candidate is not None]
    existing = [item for item in inspections if item["exists"] and item["non_roi_image_count"]]
    if not existing:
        raise FileNotFoundError("No Healthy source image folder was found.")

    exact_150 = [item for item in existing if item["non_roi_image_count"] == 150]
    if exact_150:
        return exact_150[0]
    return max(existing, key=lambda item: int(item["non_roi_image_count"]))


def allocate_healthy_images(images: list[Path], seed: int = 42) -> dict[str, list[Path]]:
    hash_groups: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        hash_groups[sha256_file(image_path)].append(image_path)

    groups = [
        sorted(paths, key=lambda path: path.as_posix().lower())
        for paths in hash_groups.values()
    ]
    rng = random.Random(seed)
    rng.shuffle(groups)

    targets = {"Training": 110, "Validation": 20, "Test": 20}
    dp: dict[tuple[int, int], list[str]] = {(0, 0): []}
    for group in groups:
        group_size = len(group)
        next_dp: dict[tuple[int, int], list[str]] = {}
        for (validation_count, test_count), assignments in dp.items():
            training_key = (validation_count, test_count)
            if training_key not in next_dp:
                next_dp[training_key] = assignments + ["Training"]

            validation_key = (validation_count + group_size, test_count)
            if validation_key[0] <= targets["Validation"] and validation_key not in next_dp:
                next_dp[validation_key] = assignments + ["Validation"]

            test_key = (validation_count, test_count + group_size)
            if test_key[1] <= targets["Test"] and test_key not in next_dp:
                next_dp[test_key] = assignments + ["Test"]
        dp = next_dp

    solution = dp.get((targets["Validation"], targets["Test"]))
    if solution is None:
        raise RuntimeError(
            "Could not allocate Healthy source images without splitting duplicate hash groups."
        )

    allocation = {"Training": [], "Validation": [], "Test": []}
    for group, split in zip(groups, solution):
        allocation[split].extend(group)

    for split, expected in targets.items():
        if len(allocation[split]) != expected:
            raise RuntimeError(
                f"Allocation mismatch for {split}: expected {expected}, got {len(allocation[split])}"
            )

    for split in allocation:
        allocation[split] = sorted(allocation[split], key=lambda path: path.as_posix().lower())
    return allocation


def write_healthy_proposal(target: Path, source_info: dict[str, object]) -> tuple[Path, Path]:
    reports_dir = target / "reports"
    csv_path = reports_dir / "final_yolo26_healthy_negative_allocation_proposal.csv"
    txt_path = reports_dir / "final_yolo26_healthy_negative_allocation_proposal.txt"

    source_folder = Path(source_info["folder"])
    images = list(source_info["images"])
    allocation = allocate_healthy_images(images, seed=42)

    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["split", "source_image", "proposed_output_image", "proposed_output_label"])
        next_id = 1
        for split in SPLITS:
            for source_image in allocation[split]:
                output_stem = f"healthy_negative_{next_id:04d}"
                writer.writerow(
                    [
                        split,
                        str(source_image),
                        f"{split}/images/{output_stem}.jpg",
                        f"{split}/labels/{output_stem}.txt",
                    ]
                )
                next_id += 1

    lines = [
        "FINAL YOLO26 HEALTHY NEGATIVE ALLOCATION PROPOSAL",
        f"Detected Healthy source folder: {source_folder}",
        f"Total Healthy source images found: {source_info['non_roi_image_count']}",
        f"All images in folder before ROI/path filtering: {source_info['all_image_count']}",
        f"Exact duplicate hash groups in source: {source_info['duplicate_hash_group_count']}",
        f"Exact duplicate extra source files: {source_info['duplicate_extra_file_count']}",
        f"Unreadable Healthy source images: {source_info['unreadable_count']}",
        "ROI-like paths excluded from this candidate: yes",
        "Fixed seed: 42",
        "",
        "Proposed split allocation",
    ]
    for split in SPLITS:
        lines.append(f"  {split}: {len(allocation[split])}")

    lines.extend(["", "Top detected source dimensions"])
    for dimension, count in source_info["top_dimensions"]:
        lines.append(f"  {dimension}: {count}")

    duplicate_groups = source_info["duplicate_groups"]
    lines.extend(["", "Source duplicate hash groups"])
    if duplicate_groups:
        for image_hash, paths in sorted(duplicate_groups.items()):
            lines.append(f"  {image_hash}")
            for path in paths:
                lines.append(f"    {path}")
    else:
        lines.append("  none")

    lines.extend(
        [
            "",
            "No Healthy negatives have been copied yet.",
            f"Proposal CSV: {csv_path}",
            "",
        ]
    )
    txt_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return csv_path, txt_path


def unique_healthy_representatives(images: list[Path]) -> tuple[list[Path], dict[str, list[Path]]]:
    hash_groups: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        hash_groups[sha256_file(image_path)].append(image_path)

    representatives: list[Path] = []
    duplicates: dict[str, list[Path]] = {}
    for image_hash, paths in sorted(hash_groups.items()):
        sorted_paths = sorted(paths, key=lambda path: path.as_posix().lower())
        representatives.append(sorted_paths[0])
        if len(sorted_paths) > 1:
            duplicates[image_hash] = sorted_paths[1:]

    return sorted(representatives, key=lambda path: path.as_posix().lower()), duplicates


def allocate_unique_healthy_images(images: list[Path], seed: int = 42) -> dict[str, list[Path]]:
    if len(images) != 142:
        raise RuntimeError(f"Expected 142 unique Healthy images, found {len(images)}")

    shuffled = list(images)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return {
        "Training": sorted(shuffled[:104], key=lambda path: path.as_posix().lower()),
        "Validation": sorted(shuffled[104:123], key=lambda path: path.as_posix().lower()),
        "Test": sorted(shuffled[123:142], key=lambda path: path.as_posix().lower()),
    }


def copy_unique_healthy_negatives(
    target: Path, source_info: dict[str, object]
) -> tuple[Path, Path, dict[str, list[Path]], dict[str, list[Path]]]:
    reports_dir = target / "reports"
    csv_path = reports_dir / "final_yolo26_healthy_negative_copied.csv"
    txt_path = reports_dir / "final_yolo26_healthy_negative_copied.txt"

    representatives, excluded_duplicates = unique_healthy_representatives(
        list(source_info["images"])
    )
    allocation = allocate_unique_healthy_images(representatives, seed=42)

    existing_outputs: list[Path] = []
    next_id = 1
    planned_rows: list[dict[str, str]] = []
    for split in SPLITS:
        images_dir = target / split / "images"
        labels_dir = target / split / "labels"
        for source_image in allocation[split]:
            stem = f"healthy_negative_{next_id:04d}"
            output_image = images_dir / f"{stem}.jpg"
            output_label = labels_dir / f"{stem}.txt"
            if output_image.exists():
                existing_outputs.append(output_image)
            if output_label.exists():
                existing_outputs.append(output_label)
            planned_rows.append(
                {
                    "split": split,
                    "source_image": str(source_image),
                    "source_sha256": sha256_file(source_image),
                    "output_image": str(output_image),
                    "output_label": str(output_label),
                }
            )
            next_id += 1

    if existing_outputs:
        listed = "\n".join(str(path) for path in existing_outputs[:20])
        raise FileExistsError(f"Refusing to overwrite existing outputs:\n{listed}")

    for row in planned_rows:
        shutil.copy2(row["source_image"], row["output_image"])
        Path(row["output_label"]).write_text("", encoding="utf-8", newline="\n")

    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "split",
                "source_image",
                "source_sha256",
                "output_image",
                "output_label",
            ],
        )
        writer.writeheader()
        writer.writerows(planned_rows)

    lines = [
        "FINAL YOLO26 HEALTHY NEGATIVE COPY REPORT",
        f"Detected Healthy source folder: {source_info['folder']}",
        f"Total source images found: {source_info['non_roi_image_count']}",
        f"Unique source image hashes used: {len(representatives)}",
        f"Excluded exact duplicate Healthy source images: {sum(len(paths) for paths in excluded_duplicates.values())}",
        "Fixed seed: 42",
        "",
        "Copied allocation",
    ]
    for split in SPLITS:
        lines.append(f"  {split}: {len(allocation[split])}")
    lines.extend(["", "Excluded duplicate source files"])
    for image_hash, paths in sorted(excluded_duplicates.items()):
        lines.append(f"  {image_hash}")
        for path in paths:
            lines.append(f"    {path}")
    lines.extend(["", f"Copy CSV: {csv_path}", ""])
    txt_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return csv_path, txt_path, allocation, excluded_duplicates


def parse_label_file(label_path: Path, split: str, rel_label: str) -> tuple[int, list[str], Counter]:
    boxes = 0
    invalid: list[str] = []
    class_ids: Counter = Counter()
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
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
            invalid.append(f"{split}/{rel_label}:{line_number}: non-numeric YOLO coordinate")
            continue
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            invalid.append(f"{split}/{rel_label}:{line_number}: center coordinate outside [0, 1]")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            invalid.append(f"{split}/{rel_label}:{line_number}: width/height outside (0, 1]")
        boxes += 1
    return boxes, invalid, class_ids


def verify_readable_image(path: Path, split: str, rel_image: str) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        return f"{split}/{rel_image}: {exc}"
    return None


def run_final_audit(
    target: Path,
    copied_allocation: dict[str, list[Path]] | None = None,
) -> tuple[Path, Path]:
    reports_dir = target / "reports"
    csv_path = reports_dir / "final_yolo26_pretraining_audit.csv"
    txt_path = reports_dir / "final_yolo26_pretraining_audit.txt"

    rows: list[dict[str, object]] = []
    all_class_ids: Counter = Counter()
    source_hash_to_splits: dict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        images_dir = target / split / "images"
        labels_dir = target / split / "labels"
        image_paths = iter_images(images_dir)
        label_paths = iter_labels(labels_dir)
        image_rel = {path.relative_to(images_dir).as_posix(): path for path in image_paths}
        label_rel = {path.relative_to(labels_dir).as_posix(): path for path in label_paths}
        expected_labels = {Path(rel).with_suffix(".txt").as_posix() for rel in image_rel}

        boxes_total = 0
        positive_images = 0
        negative_images = 0
        invalid: list[str] = []
        unreadable: list[str] = []
        class_ids: Counter = Counter()

        for rel_image, image_path in image_rel.items():
            rel_label = Path(rel_image).with_suffix(".txt").as_posix()
            if rel_label in label_rel:
                boxes, label_invalid, label_class_ids = parse_label_file(
                    label_rel[rel_label], split, rel_label
                )
                boxes_total += boxes
                invalid.extend(label_invalid)
                class_ids.update(label_class_ids)
                if boxes:
                    positive_images += 1
                else:
                    negative_images += 1
            image_error = verify_readable_image(image_path, split, rel_image)
            if image_error:
                unreadable.append(image_error)

        all_class_ids.update(class_ids)
        rows.append(
            {
                "split": split,
                "total_images": len(image_paths),
                "total_label_files": len(label_paths),
                "gingivitis_positive_images": positive_images,
                "negative_images_empty_labels": negative_images,
                "total_gingivitis_boxes": boxes_total,
                "missing_labels": len(expected_labels - set(label_rel)),
                "orphan_labels": len(set(label_rel) - expected_labels),
                "invalid_yolo_coordinates": len(invalid),
                "unreadable_images": len(unreadable),
                "remaining_class_ids": ";".join(
                    f"{class_id}:{count}" for class_id, count in sorted(class_ids.items())
                ),
            }
        )

    duplicate_groups = cross_split_duplicate_groups(collect_dataset_hashes(target))

    if copied_allocation:
        for split, source_paths in copied_allocation.items():
            for source_path in source_paths:
                source_hash_to_splits[sha256_file(source_path)].add(split)
    source_leakage = {
        image_hash: splits
        for image_hash, splits in source_hash_to_splits.items()
        if len(splits) > 1
    }

    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "FINAL YOLO26 PRE-TRAINING AUDIT",
        f"Target: {target}",
        "",
        "Per-split counts",
        "split,total_images,total_label_files,gingivitis_positive_images,"
        "negative_images_empty_labels,total_gingivitis_boxes,missing_labels,"
        "orphan_labels,invalid_yolo_coordinates,unreadable_images,remaining_class_ids",
    ]
    for row in rows:
        lines.append(
            ",".join(
                str(row[key])
                for key in [
                    "split",
                    "total_images",
                    "total_label_files",
                    "gingivitis_positive_images",
                    "negative_images_empty_labels",
                    "total_gingivitis_boxes",
                    "missing_labels",
                    "orphan_labels",
                    "invalid_yolo_coordinates",
                    "unreadable_images",
                    "remaining_class_ids",
                ]
            )
        )

    non_zero_classes = sorted(class_id for class_id in all_class_ids if class_id != "0")
    lines.extend(
        [
            "",
            f"Only class ID 0 exists in non-empty labels: {'PASS' if not non_zero_classes else 'FAIL'}",
            f"All remaining class IDs: {dict(sorted(all_class_ids.items()))}",
            f"Cross-split exact duplicate groups: {len(duplicate_groups)}",
            f"Cross-split exact duplicate check: {'PASS' if not duplicate_groups else 'FAIL'}",
            f"Healthy negative source leakage across splits: {len(source_leakage)}",
            f"Healthy negative source leakage check: {'PASS' if not source_leakage else 'FAIL'}",
            "",
            f"Audit CSV: {csv_path}",
            "",
        ]
    )
    txt_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return csv_path, txt_path


def main() -> int:
    args = parse_args()
    target = ensure_target(args.target)

    cleanup_csv = cleanup_txt = None
    remaining_duplicates = len(cross_split_duplicate_groups(collect_dataset_hashes(target)))
    if not args.proposal_only and not args.execute_unique_negatives:
        cleanup_csv, cleanup_txt, remaining_duplicates = remove_cross_split_duplicates(target)
    source_info = select_healthy_source(args.healthy_source)
    if args.execute_unique_negatives:
        copy_csv, copy_txt, copied_allocation, _excluded = copy_unique_healthy_negatives(
            target, source_info
        )
        audit_csv, audit_txt = run_final_audit(target, copied_allocation)
        print("PART 2 COMPLETE")
        print(f"Healthy negative copy CSV: {copy_csv}")
        print(f"Healthy negative copy TXT: {copy_txt}")
        print(f"Final audit CSV: {audit_csv}")
        print(f"Final audit TXT: {audit_txt}")
        return 0

    proposal_csv, proposal_txt = write_healthy_proposal(target, source_info)

    print("PART 1 COMPLETE" if not args.proposal_only else "PART 1 SKIPPED BY --proposal-only")
    print(f"Cross-split duplicate groups after cleanup: {remaining_duplicates}")
    if cleanup_csv and cleanup_txt:
        print(f"Duplicate cleanup CSV: {cleanup_csv}")
        print(f"Duplicate cleanup TXT: {cleanup_txt}")
    print()
    print("PART 2 PROPOSAL ONLY - NO HEALTHY NEGATIVES COPIED")
    print(f"Detected Healthy source folder: {source_info['folder']}")
    print(f"Total Healthy source images found: {source_info['non_roi_image_count']}")
    print(f"Exact duplicate hash groups in source: {source_info['duplicate_hash_group_count']}")
    print(f"Exact duplicate extra source files: {source_info['duplicate_extra_file_count']}")
    print("Proposed split allocation: Training=110, Validation=20, Test=20")
    print(f"Proposal CSV: {proposal_csv}")
    print(f"Proposal TXT: {proposal_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
