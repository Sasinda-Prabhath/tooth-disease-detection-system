"""Final read-only audit for the balanced gingivitis classification dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PIL import Image, ImageOps, ImageStat
except ImportError as exc:  # pragma: no cover - user-facing dependency guard.
    raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc


SPLITS = ("Training", "Validation", "Test")
CLASSES = ("Healthy", "Mild", "Moderate", "Severe")
EXPECTED_COUNTS = {
    "Training": {"Healthy": 600, "Mild": 600, "Moderate": 600, "Severe": 600},
    "Validation": {"Healthy": 30, "Mild": 1080, "Moderate": 755, "Severe": 320},
    "Test": {"Healthy": 35, "Mild": 1206, "Moderate": 713, "Severe": 221},
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_TOP_LEVEL = set(SPLITS) | {"healthy_topup", "reports"}
MIN_EXTREME_SIDE = 64
MIN_SUSPICIOUS_SIDE = 128
MAX_SUSPICIOUS_ASPECT_RATIO = 4.0
NEAR_DUPLICATE_DISTANCE = 3
MAX_NEAR_DUPLICATE_PAIRS = 10000


@dataclass
class ImageRecord:
    split: str
    class_name: str
    path: Path
    rel_path: str
    filename: str
    source_stem: str
    size_bytes: int = 0
    extension_supported: bool = True
    zero_byte: bool = False
    readable: bool = False
    width: int = 0
    height: int = 0
    sha256: str = ""
    dhash: int | None = None
    issue: str = ""
    dimension_flags: list[str] = field(default_factory=list)


@dataclass
class AuditResult:
    records: list[ImageRecord]
    unsupported_files: list[Path]
    unexpected_dirs: list[Path]
    unexpected_files: list[Path]
    counts: dict[str, dict[str, int]]
    split_totals: dict[str, int]
    duplicate_groups: list[list[ImageRecord]]
    cross_split_duplicate_groups: list[list[ImageRecord]]
    source_leakage: dict[tuple[str, str], dict[str, list[ImageRecord]]]
    synthetic_by_split: dict[str, list[ImageRecord]]
    duplicate_filenames: dict[str, list[ImageRecord]]
    near_duplicate_pairs: list[tuple[ImageRecord, ImageRecord, int]]
    readiness_status: str
    blockers: list[str]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_stem_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if "_roi_" in stem:
        base = stem.split("_roi_", 1)[0]
    else:
        base = stem
    if base.startswith("healthy_syn_"):
        return base
    for class_name in CLASSES:
        suffix = f"_{class_name}"
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    get_pixels = getattr(gray, "get_flattened_data", gray.getdata)
    pixels = list(get_pixels())
    value = 0
    bit = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            if pixels[offset + col] > pixels[offset + col + 1]:
                value |= 1 << bit
            bit += 1
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def inspect_image(record: ImageRecord) -> None:
    try:
        record.size_bytes = record.path.stat().st_size
    except OSError as exc:
        record.issue = f"cannot stat file: {exc}"
        return

    record.zero_byte = record.size_bytes == 0
    record.extension_supported = record.path.suffix.lower() in IMAGE_EXTENSIONS
    if record.zero_byte:
        record.issue = "zero-byte file"
        return
    if not record.extension_supported:
        record.issue = "unsupported file type"
        return

    try:
        record.sha256 = file_sha256(record.path)
        with Image.open(record.path) as image:
            image.verify()
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            record.width, record.height = image.size
            record.dhash = dhash(image)
        record.readable = True
        flag_dimensions(record)
    except Exception as exc:  # pragma: no cover - defensive for damaged files.
        record.issue = f"unreadable or corrupted image: {exc}"


def flag_dimensions(record: ImageRecord) -> None:
    if record.width <= 0 or record.height <= 0:
        record.dimension_flags.append("invalid dimensions")
        return
    min_side = min(record.width, record.height)
    max_side = max(record.width, record.height)
    aspect_ratio = max_side / min_side
    if min_side < MIN_EXTREME_SIDE:
        record.dimension_flags.append(f"extremely small side < {MIN_EXTREME_SIDE}px")
    elif min_side < MIN_SUSPICIOUS_SIDE:
        record.dimension_flags.append(f"suspicious small side < {MIN_SUSPICIOUS_SIDE}px")
    if aspect_ratio > MAX_SUSPICIOUS_ASPECT_RATIO:
        record.dimension_flags.append(f"unusual aspect ratio {aspect_ratio:.2f}")


def collect_records(dataset_dir: Path) -> tuple[list[ImageRecord], list[Path], list[Path], list[Path]]:
    records: list[ImageRecord] = []
    unsupported_files: list[Path] = []
    unexpected_dirs: list[Path] = []
    unexpected_files: list[Path] = []

    for child in dataset_dir.iterdir() if dataset_dir.exists() else []:
        if child.name not in EXPECTED_TOP_LEVEL:
            if child.is_dir():
                unexpected_dirs.append(child)
            else:
                unexpected_files.append(child)

    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            if child.is_dir() and child.name not in CLASSES:
                unexpected_dirs.append(child)
            elif child.is_file():
                unexpected_files.append(child)

        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir(), key=lambda item: item.name.lower()):
                if path.is_dir():
                    unexpected_dirs.append(path)
                    continue
                rel_path = path.relative_to(dataset_dir).as_posix()
                record = ImageRecord(
                    split=split,
                    class_name=class_name,
                    path=path,
                    rel_path=rel_path,
                    filename=path.name,
                    source_stem=source_stem_from_filename(path.name),
                )
                inspect_image(record)
                if not record.extension_supported:
                    unsupported_files.append(path)
                records.append(record)

    return records, unsupported_files, unexpected_dirs, unexpected_files


def group_exact_duplicates(records: list[ImageRecord]) -> tuple[list[list[ImageRecord]], list[list[ImageRecord]]]:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.sha256:
            by_hash[record.sha256].append(record)

    duplicate_groups = [items for items in by_hash.values() if len(items) > 1]
    cross_split_groups = [items for items in duplicate_groups if len({item.split for item in items}) > 1]
    return duplicate_groups, cross_split_groups


def find_source_leakage(records: list[ImageRecord]) -> dict[tuple[str, str], dict[str, list[ImageRecord]]]:
    by_source: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.source_stem:
            by_source[record.source_stem].append(record)

    leakage: dict[tuple[str, str], dict[str, list[ImageRecord]]] = {}
    for source_stem, items in by_source.items():
        by_split: dict[str, list[ImageRecord]] = defaultdict(list)
        for item in items:
            by_split[item.split].append(item)
        present_splits = sorted(by_split)
        if len(present_splits) <= 1:
            continue
        for left_index, left in enumerate(present_splits):
            for right in present_splits[left_index + 1 :]:
                leakage[(source_stem, f"{left}<->{right}")] = {left: by_split[left], right: by_split[right]}
    return leakage


def find_duplicate_filenames(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    by_name: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_name[record.filename].append(record)
    return {name: items for name, items in by_name.items() if len(items) > 1}


def find_near_duplicates(records: list[ImageRecord], max_pairs: int) -> list[tuple[ImageRecord, ImageRecord, int]]:
    by_split = {split: [item for item in records if item.split == split and item.dhash is not None] for split in SPLITS}
    pairs: list[tuple[ImageRecord, ImageRecord, int]] = []

    for left_split, right_split in (("Training", "Validation"), ("Training", "Test"), ("Validation", "Test")):
        for left in by_split[left_split]:
            for right in by_split[right_split]:
                distance = hamming_distance(left.dhash or 0, right.dhash or 0)
                if distance <= NEAR_DUPLICATE_DISTANCE:
                    pairs.append((left, right, distance))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def dimension_summary(records: list[ImageRecord]) -> dict[str, float | int]:
    readable = [record for record in records if record.readable]
    if not readable:
        return {
            "min_width": 0,
            "min_height": 0,
            "max_width": 0,
            "max_height": 0,
            "avg_width": 0.0,
            "avg_height": 0.0,
            "flagged_dimensions": 0,
        }
    return {
        "min_width": min(record.width for record in readable),
        "min_height": min(record.height for record in readable),
        "max_width": max(record.width for record in readable),
        "max_height": max(record.height for record in readable),
        "avg_width": sum(record.width for record in readable) / len(readable),
        "avg_height": sum(record.height for record in readable) / len(readable),
        "flagged_dimensions": sum(1 for record in readable if record.dimension_flags),
    }


def audit_dataset(args: argparse.Namespace) -> AuditResult:
    dataset_dir = Path(args.dataset_dir)
    records, unsupported_files, unexpected_dirs, unexpected_files = collect_records(dataset_dir)

    counts = {split: {class_name: 0 for class_name in CLASSES} for split in SPLITS}
    for record in records:
        if record.extension_supported:
            counts[record.split][record.class_name] += 1
    split_totals = {split: sum(counts[split].values()) for split in SPLITS}

    duplicate_groups, cross_split_duplicate_groups = group_exact_duplicates(records)
    source_leakage = find_source_leakage(records)
    duplicate_filenames = find_duplicate_filenames(records)
    near_duplicate_pairs = find_near_duplicates(records, args.max_near_duplicate_pairs)
    synthetic_by_split = {
        split: [record for record in records if record.split == split and record.filename.startswith("healthy_syn_")]
        for split in SPLITS
    }

    blockers = readiness_blockers(
        records=records,
        counts=counts,
        unsupported_files=unsupported_files,
        unexpected_dirs=unexpected_dirs,
        unexpected_files=unexpected_files,
        cross_split_duplicate_groups=cross_split_duplicate_groups,
        source_leakage=source_leakage,
        synthetic_by_split=synthetic_by_split,
        duplicate_filenames=duplicate_filenames,
    )
    return AuditResult(
        records=records,
        unsupported_files=unsupported_files,
        unexpected_dirs=unexpected_dirs,
        unexpected_files=unexpected_files,
        counts=counts,
        split_totals=split_totals,
        duplicate_groups=duplicate_groups,
        cross_split_duplicate_groups=cross_split_duplicate_groups,
        source_leakage=source_leakage,
        synthetic_by_split=synthetic_by_split,
        duplicate_filenames=duplicate_filenames,
        near_duplicate_pairs=near_duplicate_pairs,
        readiness_status="READY FOR TRAINING" if not blockers else "NOT READY FOR TRAINING",
        blockers=blockers,
    )


def readiness_blockers(
    records: list[ImageRecord],
    counts: dict[str, dict[str, int]],
    unsupported_files: list[Path],
    unexpected_dirs: list[Path],
    unexpected_files: list[Path],
    cross_split_duplicate_groups: list[list[ImageRecord]],
    source_leakage: dict[tuple[str, str], dict[str, list[ImageRecord]]],
    synthetic_by_split: dict[str, list[ImageRecord]],
    duplicate_filenames: dict[str, list[ImageRecord]],
) -> list[str]:
    blockers: list[str] = []
    for split in SPLITS:
        for class_name in CLASSES:
            expected = EXPECTED_COUNTS[split][class_name]
            actual = counts[split][class_name]
            if actual != expected:
                blockers.append(f"{split}/{class_name} count is {actual}, expected {expected}")

    unreadable = [record for record in records if record.extension_supported and (record.zero_byte or not record.readable)]
    if unreadable:
        blockers.append(f"{len(unreadable)} unreadable/corrupted/zero-byte image files")
    if unsupported_files:
        blockers.append(f"{len(unsupported_files)} unsupported files inside class folders")
    if unexpected_dirs or unexpected_files:
        blockers.append(f"{len(unexpected_dirs)} unexpected folders and {len(unexpected_files)} unexpected files in dataset structure")
    if cross_split_duplicate_groups:
        blockers.append(f"{len(cross_split_duplicate_groups)} exact duplicate hash groups cross dataset splits")
    if source_leakage:
        blockers.append(f"{len(source_leakage)} source-stem overlaps across dataset splits")
    if synthetic_by_split["Validation"] or synthetic_by_split["Test"]:
        blockers.append("synthetic healthy_syn_* files exist outside Training")

    cross_split_name_collisions = [
        items for items in duplicate_filenames.values() if len({item.split for item in items}) > 1
    ]
    if cross_split_name_collisions:
        blockers.append(f"{len(cross_split_name_collisions)} duplicate filename collisions across splits")
    return blockers


def write_counts_csv(path: Path, result: AuditResult) -> None:
    fieldnames = [
        "split",
        "class",
        "image_count",
        "expected_count",
        "count_ok",
        "readable_count",
        "unreadable_or_corrupt_count",
        "zero_byte_count",
        "unsupported_file_count",
        "min_width",
        "min_height",
        "max_width",
        "max_height",
        "avg_width",
        "avg_height",
        "dimension_warning_count",
        "synthetic_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split in SPLITS:
            for class_name in CLASSES:
                items = [record for record in result.records if record.split == split and record.class_name == class_name]
                dims = dimension_summary(items)
                expected = EXPECTED_COUNTS[split][class_name]
                writer.writerow(
                    {
                        "split": split,
                        "class": class_name,
                        "image_count": result.counts[split][class_name],
                        "expected_count": expected,
                        "count_ok": result.counts[split][class_name] == expected,
                        "readable_count": sum(1 for item in items if item.readable),
                        "unreadable_or_corrupt_count": sum(1 for item in items if item.extension_supported and not item.readable),
                        "zero_byte_count": sum(1 for item in items if item.zero_byte),
                        "unsupported_file_count": sum(1 for item in items if not item.extension_supported),
                        "min_width": dims["min_width"],
                        "min_height": dims["min_height"],
                        "max_width": dims["max_width"],
                        "max_height": dims["max_height"],
                        "avg_width": f"{dims['avg_width']:.2f}",
                        "avg_height": f"{dims['avg_height']:.2f}",
                        "dimension_warning_count": dims["flagged_dimensions"],
                        "synthetic_count": sum(1 for item in items if item.filename.startswith("healthy_syn_")),
                    }
                )


def write_duplicates_csv(path: Path, result: AuditResult) -> None:
    fieldnames = ["duplicate_type", "sha256", "count", "splits", "classes", "files", "possible_leakage"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for group in result.duplicate_groups:
            splits = sorted({item.split for item in group})
            writer.writerow(
                {
                    "duplicate_type": "exact_hash",
                    "sha256": group[0].sha256,
                    "count": len(group),
                    "splits": ";".join(splits),
                    "classes": ";".join(sorted({item.class_name for item in group})),
                    "files": ";".join(item.rel_path for item in group),
                    "possible_leakage": len(splits) > 1,
                }
            )

        for filename, items in sorted(result.duplicate_filenames.items()):
            splits = sorted({item.split for item in items})
            writer.writerow(
                {
                    "duplicate_type": "filename_collision",
                    "sha256": "",
                    "count": len(items),
                    "splits": ";".join(splits),
                    "classes": ";".join(sorted({item.class_name for item in items})),
                    "files": ";".join(item.rel_path for item in items),
                    "possible_leakage": len(splits) > 1,
                }
            )


def write_leakage_csv(path: Path, result: AuditResult) -> None:
    fieldnames = ["leakage_type", "key", "split_pair", "count", "files", "details"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for group in result.cross_split_duplicate_groups:
            split_pair = "<->".join(sorted({item.split for item in group}))
            writer.writerow(
                {
                    "leakage_type": "exact_duplicate_cross_split",
                    "key": group[0].sha256,
                    "split_pair": split_pair,
                    "count": len(group),
                    "files": ";".join(item.rel_path for item in group),
                    "details": "same file hash appears in multiple splits",
                }
            )

        for (source_stem, split_pair), by_split in sorted(result.source_leakage.items()):
            items = [item for split_items in by_split.values() for item in split_items]
            writer.writerow(
                {
                    "leakage_type": "source_stem_cross_split",
                    "key": source_stem,
                    "split_pair": split_pair,
                    "count": len(items),
                    "files": ";".join(item.rel_path for item in items),
                    "details": "ROI crops from the same source image appear in multiple splits",
                }
            )

        for split in ("Validation", "Test"):
            for item in result.synthetic_by_split[split]:
                writer.writerow(
                    {
                        "leakage_type": "synthetic_in_non_training_split",
                        "key": item.filename,
                        "split_pair": split,
                        "count": 1,
                        "files": item.rel_path,
                        "details": "healthy_syn_* should exist only in Training/Healthy",
                    }
                )


def write_near_duplicates_csv(path: Path, result: AuditResult) -> None:
    fieldnames = [
        "file_a",
        "split_a",
        "class_a",
        "source_stem_a",
        "file_b",
        "split_b",
        "class_b",
        "source_stem_b",
        "dhash_distance",
        "warning",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for left, right, distance in result.near_duplicate_pairs:
            writer.writerow(
                {
                    "file_a": left.rel_path,
                    "split_a": left.split,
                    "class_a": left.class_name,
                    "source_stem_a": left.source_stem,
                    "file_b": right.rel_path,
                    "split_b": right.split,
                    "class_b": right.class_name,
                    "source_stem_b": right.source_stem,
                    "dhash_distance": distance,
                    "warning": "cross-split perceptual near-duplicate candidate",
                }
            )


def write_audit_txt(path: Path, result: AuditResult) -> None:
    records = result.records
    total = len([item for item in records if item.extension_supported])
    unreadable = [item for item in records if item.extension_supported and not item.readable]
    dims = dimension_summary(records)
    train_healthy = [item for item in records if item.split == "Training" and item.class_name == "Healthy"]
    synthetic_train_healthy = [item for item in train_healthy if item.filename.startswith("healthy_syn_")]
    real_train_healthy = [item for item in train_healthy if not item.filename.startswith("healthy_syn_")]
    synthetic_pct = (len(synthetic_train_healthy) / len(train_healthy) * 100) if train_healthy else 0.0
    exact_duplicate_file_count = sum(len(group) for group in result.duplicate_groups)
    cross_split_exact_file_count = sum(len(group) for group in result.cross_split_duplicate_groups)

    lines = [
        "Final classification dataset audit",
        "==================================",
        "",
        "Class counts:",
    ]
    for split in SPLITS:
        counts = ", ".join(f"{class_name}={result.counts[split][class_name]}" for class_name in CLASSES)
        lines.append(f"- {split}: {counts}; total={result.split_totals[split]}")

    lines.extend(
        [
            "",
            f"total dataset images: {total}",
            f"unreadable/corrupted/zero-byte supported files: {len(unreadable)}",
            f"unsupported files in class folders: {len(result.unsupported_files)}",
            f"unexpected folders: {len(result.unexpected_dirs)}",
            f"unexpected files: {len(result.unexpected_files)}",
            "",
            "Image dimensions:",
            f"- min width/height: {dims['min_width']} x {dims['min_height']}",
            f"- max width/height: {dims['max_width']} x {dims['max_height']}",
            f"- average width/height: {dims['avg_width']:.2f} x {dims['avg_height']:.2f}",
            f"- suspicious dimension warnings: {dims['flagged_dimensions']}",
            "",
            "Exact duplicates:",
            f"- duplicate hash groups: {len(result.duplicate_groups)}",
            f"- duplicate files in duplicate groups: {exact_duplicate_file_count}",
            f"- cross-split duplicate hash groups: {len(result.cross_split_duplicate_groups)}",
            f"- cross-split duplicate files: {cross_split_exact_file_count}",
            "",
            "Source-stem leakage:",
            f"- total cross-split source-stem overlaps: {len(result.source_leakage)}",
            f"- Training<->Validation: {count_leakage_pairs(result, 'Training<->Validation')}",
            f"- Training<->Test: {count_leakage_pairs(result, 'Training<->Test')}",
            f"- Test<->Validation: {count_leakage_pairs(result, 'Test<->Validation')}",
            f"- Validation<->Test: {count_leakage_pairs(result, 'Validation<->Test')}",
            "",
            "Synthetic Healthy safety:",
            f"- synthetic images in Training: {len(result.synthetic_by_split['Training'])}",
            f"- synthetic images in Validation: {len(result.synthetic_by_split['Validation'])}",
            f"- synthetic images in Test: {len(result.synthetic_by_split['Test'])}",
            f"- synthetic Training/Healthy count: {len(synthetic_train_healthy)}",
            f"- real/expert Training/Healthy count: {len(real_train_healthy)}",
            f"- synthetic percentage of Training/Healthy: {synthetic_pct:.2f}%",
            "",
            "Filename duplication:",
            f"- duplicate filename groups: {len(result.duplicate_filenames)}",
            f"- cross-split duplicate filename groups: {sum(1 for items in result.duplicate_filenames.values() if len({item.split for item in items}) > 1)}",
            "",
            "Near-duplicates:",
            f"- cross-split near-duplicate warnings: {len(result.near_duplicate_pairs)}",
            f"- dHash warning threshold: <= {NEAR_DUPLICATE_DISTANCE}",
            "",
            f"FINAL STATUS: {result.readiness_status}",
        ]
    )

    if result.blockers:
        lines.append("")
        lines.append("Fix before training:")
        for blocker in result.blockers:
            lines.append(f"- {blocker}")

    lines.append("")
    lines.append("Audit is read-only. No dataset images were modified, moved, renamed, resized, overwritten, or deleted.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_leakage_pairs(result: AuditResult, split_pair: str) -> int:
    return sum(1 for (_source, pair) in result.source_leakage if pair == split_pair)


def write_reports(result: AuditResult, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_audit_txt(reports_dir / "final_dataset_audit.txt", result)
    write_counts_csv(reports_dir / "final_dataset_counts.csv", result)
    write_duplicates_csv(reports_dir / "final_dataset_duplicates.csv", result)
    write_leakage_csv(reports_dir / "final_dataset_leakage.csv", result)
    write_near_duplicates_csv(reports_dir / "final_dataset_near_duplicates.csv", result)


def print_summary(result: AuditResult) -> None:
    print("Final classification dataset audit")
    print("==================================")
    for split in SPLITS:
        counts = ", ".join(f"{class_name}={result.counts[split][class_name]}" for class_name in CLASSES)
        print(f"{split}: {counts}; total={result.split_totals[split]}")
    print(f"total dataset images: {sum(result.split_totals.values())}")
    print(f"unreadable/corrupted/zero-byte supported files: {sum(1 for item in result.records if item.extension_supported and not item.readable)}")
    print(f"unsupported files in class folders: {len(result.unsupported_files)}")
    print(f"exact duplicate hash groups: {len(result.duplicate_groups)}")
    print(f"cross-split exact duplicate groups: {len(result.cross_split_duplicate_groups)}")
    print(f"source-stem cross-split overlaps: {len(result.source_leakage)}")
    print(f"synthetic Training/Healthy: {sum(1 for item in result.records if item.split == 'Training' and item.class_name == 'Healthy' and item.filename.startswith('healthy_syn_'))}")
    print(f"real/expert Training/Healthy: {sum(1 for item in result.records if item.split == 'Training' and item.class_name == 'Healthy' and not item.filename.startswith('healthy_syn_'))}")
    print(f"synthetic in Validation: {len(result.synthetic_by_split['Validation'])}")
    print(f"synthetic in Test: {len(result.synthetic_by_split['Test'])}")
    print(f"cross-split near-duplicate warnings: {len(result.near_duplicate_pairs)}")
    print(f"FINAL STATUS: {result.readiness_status}")
    if result.blockers:
        print("Fix before training:")
        for blocker in result.blockers:
            print(f"- {blocker}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(default_dataset_dir(root)))
    parser.add_argument("--reports-dir", default=str(default_dataset_dir(root) / "reports"))
    parser.add_argument("--max-near-duplicate-pairs", type=int, default=MAX_NEAR_DUPLICATE_PAIRS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_dataset(args)
    write_reports(result, Path(args.reports_dir))
    print_summary(result)


if __name__ == "__main__":
    main()
