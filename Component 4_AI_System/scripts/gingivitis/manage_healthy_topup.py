"""Manage Healthy top-up candidates for the gingivitis classifier dataset.

This script keeps candidate Healthy ROIs in the healthy_topup workflow until a
human reviewer manually places accepted files in healthy_topup/approved.

Commands:
  scan  - inspect candidates and write review reports; no Training files copied
  merge - copy approved files into Training/Healthy, with --dry-run support
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - user-facing dependency guard.
    raise SystemExit("Pillow is required to validate images. Install it with: pip install pillow") from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_HEALTHY_COUNT = 600
DEFAULT_MIN_SIZE = 16
CLASSIFICATION_BALANCED_RELATIVE = Path("datasets") / "gingivitis" / "classification_balanced"


@dataclass
class ImageRecord:
    path: Path
    group: str
    filename: str
    suffix: str
    readable: bool
    width: int | None
    height: int | None
    sha256: str | None
    status: str
    reason: str


@dataclass
class MergeAction:
    source: Path
    destination: Path | None
    action: str
    reason: str
    sha256: str | None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_balanced_root(root: Path) -> Path:
    return root / CLASSIFICATION_BALANCED_RELATIVE


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path, group: str, min_size: int) -> ImageRecord:
    width: int | None = None
    height: int | None = None
    digest: str | None = None
    readable = False
    status = "candidate"
    reason = "ready for manual review"

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        readable = True
        digest = sha256_file(path)
    except OSError as exc:
        status = "reject"
        reason = f"unreadable image: {exc}"
        return ImageRecord(path, group, path.name, path.suffix.lower(), readable, width, height, digest, status, reason)

    if width is None or height is None or width < min_size or height < min_size:
        status = "reject"
        reason = f"image smaller than minimum {min_size}x{min_size}"

    return ImageRecord(path, group, path.name, path.suffix.lower(), readable, width, height, digest, status, reason)


def scan_group(folder: Path, group: str, min_size: int) -> list[ImageRecord]:
    return [inspect_image(path, group, min_size) for path in image_files(folder)]


def mark_duplicates(records: list[ImageRecord]) -> None:
    filename_groups: dict[str, list[ImageRecord]] = defaultdict(list)
    hash_groups: dict[str, list[ImageRecord]] = defaultdict(list)

    for record in records:
        filename_groups[record.filename.lower()].append(record)
        if record.sha256:
            hash_groups[record.sha256].append(record)

    for same_name_records in filename_groups.values():
        if len(same_name_records) > 1:
            for record in same_name_records:
                append_reason(record, "duplicate filename in candidate folders")

    for same_hash_records in hash_groups.values():
        if len(same_hash_records) > 1:
            for record in same_hash_records:
                append_reason(record, "exact duplicate image hash in candidate folders")


def append_reason(record: ImageRecord, extra_reason: str) -> None:
    if record.status == "candidate":
        record.status = "review"
    if record.reason:
        record.reason = f"{record.reason}; {extra_reason}"
    else:
        record.reason = extra_reason


def ensure_workflow_structure(balanced_root: Path) -> None:
    for folder_name in ("real", "synthetic", "approved", "rejected"):
        (balanced_root / "healthy_topup" / folder_name).mkdir(parents=True, exist_ok=True)
    (balanced_root / "reports").mkdir(parents=True, exist_ok=True)


def training_healthy_dir(balanced_root: Path) -> Path:
    return balanced_root / "Training" / "Healthy"


def reports_dir(balanced_root: Path) -> Path:
    return balanced_root / "reports"


def topup_dir(balanced_root: Path, name: str) -> Path:
    return balanced_root / "healthy_topup" / name


def count_images(folder: Path) -> int:
    return len(image_files(folder))


def write_candidate_report(path: Path, records: list[ImageRecord]) -> None:
    fieldnames = [
        "group",
        "filename",
        "path",
        "readable",
        "width",
        "height",
        "sha256",
        "status",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record_row(record))


def write_simple_image_report(path: Path, records: list[ImageRecord]) -> None:
    fieldnames = ["group", "filename", "path", "readable", "width", "height", "sha256", "status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record_row(record))


def record_row(record: ImageRecord) -> dict[str, str | int | bool | None]:
    return {
        "group": record.group,
        "filename": record.filename,
        "path": str(record.path),
        "readable": record.readable,
        "width": record.width,
        "height": record.height,
        "sha256": record.sha256,
        "status": record.status,
        "reason": record.reason,
    }


def write_summary(
    balanced_root: Path,
    candidate_records: list[ImageRecord],
    approved_records: list[ImageRecord],
    rejected_records: list[ImageRecord],
    final_healthy_count_after_merge: int | None,
    target: int,
    min_size: int,
) -> None:
    current_healthy = count_images(training_healthy_dir(balanced_root))
    required_remaining = max(0, target - current_healthy)
    real_count = sum(1 for record in candidate_records if record.group == "real")
    synthetic_count = sum(1 for record in candidate_records if record.group == "synthetic")

    lines = [
        "Healthy top-up workflow summary",
        "================================",
        f"Balanced dataset path: {balanced_root}",
        f"Minimum accepted image size: {min_size}x{min_size}",
        "",
        f"Current Healthy Training count = {current_healthy}",
        f"Target count = {target}",
        f"Required remaining count = {required_remaining}",
        f"Real candidate count = {real_count}",
        f"Synthetic candidate count = {synthetic_count}",
        f"Approved count = {len(approved_records)}",
        f"Rejected count = {len(rejected_records)}",
        f"Final Healthy count after merge = {format_unknown(final_healthy_count_after_merge)}",
        "",
        "Rules:",
        "  No candidate image is automatically approved.",
        "  Only files manually placed in healthy_topup/approved are eligible for merge.",
        "  Validation and Test are never modified.",
        "  Existing Training/Healthy files are never overwritten.",
    ]
    (reports_dir(balanced_root) / "healthy_topup_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_unknown(value: int | None) -> str:
    return "not merged yet" if value is None else str(value)


def scan_workflow(balanced_root: Path, target: int, min_size: int, final_healthy_count_after_merge: int | None = None) -> None:
    ensure_workflow_structure(balanced_root)

    candidate_records = scan_group(topup_dir(balanced_root, "real"), "real", min_size)
    candidate_records.extend(scan_group(topup_dir(balanced_root, "synthetic"), "synthetic", min_size))
    mark_duplicates(candidate_records)

    approved_records = scan_group(topup_dir(balanced_root, "approved"), "approved", min_size)
    rejected_records = scan_group(topup_dir(balanced_root, "rejected"), "rejected", min_size)

    report_root = reports_dir(balanced_root)
    write_candidate_report(report_root / "healthy_candidates.csv", candidate_records)
    write_simple_image_report(report_root / "healthy_approved.csv", approved_records)
    write_simple_image_report(report_root / "healthy_rejected.csv", rejected_records)
    write_summary(
        balanced_root,
        candidate_records,
        approved_records,
        rejected_records,
        final_healthy_count_after_merge,
        target,
        min_size,
    )

    print_scan_summary(balanced_root, candidate_records, approved_records, rejected_records, target)


def print_scan_summary(
    balanced_root: Path,
    candidate_records: list[ImageRecord],
    approved_records: list[ImageRecord],
    rejected_records: list[ImageRecord],
    target: int,
) -> None:
    current_healthy = count_images(training_healthy_dir(balanced_root))
    print("Healthy top-up scan complete")
    print("============================")
    print(f"Balanced dataset path: {balanced_root}")
    print(f"Current Healthy Training count: {current_healthy}")
    print(f"Target Healthy Training count: {target}")
    print(f"Required remaining count: {max(0, target - current_healthy)}")
    print(f"Real candidates: {sum(1 for record in candidate_records if record.group == 'real')}")
    print(f"Synthetic candidates: {sum(1 for record in candidate_records if record.group == 'synthetic')}")
    print(f"Approved folder count: {len(approved_records)}")
    print(f"Rejected folder count: {len(rejected_records)}")
    print("No images were copied into Training/Healthy.")


def existing_hashes(folder: Path) -> set[str]:
    hashes: set[str] = set()
    for path in image_files(folder):
        try:
            hashes.add(sha256_file(path))
        except OSError:
            continue
    return hashes


def safe_destination(destination_dir: Path, source_name: str) -> Path:
    candidate = destination_dir / source_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        renamed = destination_dir / f"{stem}_topup_{index:03d}{suffix}"
        if not renamed.exists():
            return renamed
        index += 1


def plan_merge(balanced_root: Path, target: int, min_size: int) -> tuple[list[MergeAction], int]:
    destination_dir = training_healthy_dir(balanced_root)
    current_count = count_images(destination_dir)
    needed = max(0, target - current_count)
    approved_records = scan_group(topup_dir(balanced_root, "approved"), "approved", min_size)

    training_hashes = existing_hashes(destination_dir)
    approved_seen_hashes: set[str] = set()
    actions: list[MergeAction] = []
    accepted_count = 0

    for record in approved_records:
        if accepted_count >= needed:
            actions.append(MergeAction(record.path, None, "skip", "target count already reached", record.sha256))
            continue
        if not record.readable:
            actions.append(MergeAction(record.path, None, "skip", record.reason, record.sha256))
            continue
        if record.status == "reject":
            actions.append(MergeAction(record.path, None, "skip", record.reason, record.sha256))
            continue
        if record.sha256 in training_hashes:
            actions.append(MergeAction(record.path, None, "skip", "exact duplicate of existing Training/Healthy image", record.sha256))
            continue
        if record.sha256 in approved_seen_hashes:
            actions.append(MergeAction(record.path, None, "skip", "exact duplicate within approved folder", record.sha256))
            continue

        destination = safe_destination(destination_dir, record.filename)
        actions.append(MergeAction(record.path, destination, "copy", "approved for merge", record.sha256))
        if record.sha256:
            approved_seen_hashes.add(record.sha256)
            training_hashes.add(record.sha256)
        accepted_count += 1

    return actions, current_count + accepted_count


def write_merge_plan(path: Path, actions: list[MergeAction], dry_run: bool) -> None:
    fieldnames = ["dry_run", "action", "reason", "source_path", "destination_path", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for action in actions:
            writer.writerow(
                {
                    "dry_run": dry_run,
                    "action": action.action,
                    "reason": action.reason,
                    "source_path": str(action.source),
                    "destination_path": "" if action.destination is None else str(action.destination),
                    "sha256": action.sha256,
                }
            )


def merge_approved(balanced_root: Path, target: int, min_size: int, dry_run: bool) -> None:
    ensure_workflow_structure(balanced_root)
    actions, final_count = plan_merge(balanced_root, target, min_size)
    copy_actions = [action for action in actions if action.action == "copy"]

    if not dry_run:
        for action in copy_actions:
            if action.destination is None:
                continue
            if action.destination.exists():
                raise RuntimeError(f"Refusing to overwrite existing file: {action.destination}")
            shutil.copy2(action.source, action.destination)
        final_count = count_images(training_healthy_dir(balanced_root))

    write_merge_plan(reports_dir(balanced_root) / "healthy_merge_plan.csv", actions, dry_run)
    scan_workflow(balanced_root, target, min_size, final_count)

    print("")
    print("Healthy approved merge")
    print("======================")
    print(f"Mode: {'dry-run' if dry_run else 'copy'}")
    print(f"Approved files eligible to copy: {len(copy_actions)}")
    print(f"Final Healthy count after merge: {final_count}")
    if dry_run:
        print("Dry-run only: no files were copied.")
    else:
        print("Approved files were copied without overwriting existing files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Healthy top-up candidates and approved merges.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root.")
    parser.add_argument("--balanced-root", type=Path, default=None, help="classification_balanced folder.")
    parser.add_argument("--target", type=int, default=TARGET_HEALTHY_COUNT, help="Target Training/Healthy count.")
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE, help="Reject images smaller than this size.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("scan", help="Scan top-up folders and write review reports.")
    merge_parser = subparsers.add_parser("merge", help="Copy approved images into Training/Healthy.")
    merge_parser.add_argument("--dry-run", action="store_true", help="Preview merge actions without copying files.")

    args = parser.parse_args()
    root = args.project_root.resolve()
    balanced_root = args.balanced_root.resolve() if args.balanced_root else default_balanced_root(root).resolve()

    if args.command == "scan":
        scan_workflow(balanced_root, args.target, args.min_size)
    elif args.command == "merge":
        merge_approved(balanced_root, args.target, args.min_size, args.dry_run)


if __name__ == "__main__":
    main()
