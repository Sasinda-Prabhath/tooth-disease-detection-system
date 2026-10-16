"""Prepare a balanced gingivitis classifier dataset without oversampling.

The source classification_levels dataset is treated as read-only. New files are
written only under datasets/gingivitis/classification_balanced.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - user-facing dependency guard.
    raise SystemExit("Pillow is required to validate images. Install it with: pip install pillow") from exc


SPLITS = ("Training", "Validation", "Test")
CLASSES = ("Healthy", "Mild", "Moderate", "Severe")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TRAINING_TARGET_PER_CLASS = 600
RANDOM_SEED = 42


@dataclass
class CopyPlan:
    selected: dict[str, dict[str, list[Path]]] = field(
        default_factory=lambda: {split: {class_name: [] for class_name in CLASSES} for split in SPLITS}
    )
    unreadable_images: list[Path] = field(default_factory=list)
    source_duplicate_filenames: dict[str, list[str]] = field(default_factory=dict)
    output_duplicate_filenames: dict[str, list[str]] = field(default_factory=dict)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_source_root(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_levels"


def default_output_root(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced"


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        [file_path for file_path in path.iterdir() if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda file_path: file_path.name.lower(),
    )


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except OSError:
        return False


def ensure_output_structure(output_root: Path) -> None:
    for split in SPLITS:
        for class_name in CLASSES:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)
    for folder_name in ("real", "synthetic", "approved", "rejected"):
        (output_root / "healthy_topup" / folder_name).mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)


def duplicate_names(paths: list[Path]) -> list[str]:
    counts = Counter(path.name.lower() for path in paths)
    return sorted([name for name, count in counts.items() if count > 1])


def source_image_stem(roi_path: Path) -> str:
    stem = roi_path.stem
    for class_name in CLASSES:
        marker = f"_{class_name}_roi_"
        if marker in stem:
            return stem.split(marker, maxsplit=1)[0]
    return stem


def build_copy_plan(source_root: Path, seed: int, target: int) -> CopyPlan:
    rng = random.Random(seed)
    plan = CopyPlan()

    for split in SPLITS:
        split_paths: list[Path] = []
        for class_name in CLASSES:
            files = image_files(source_root / split / class_name)
            split_paths.extend(files)

            unreadable = [path for path in files if not validate_image(path)]
            plan.unreadable_images.extend(unreadable)
            readable_files = [path for path in files if path not in set(unreadable)]

            if split == "Training" and class_name != "Healthy" and len(readable_files) > target:
                plan.selected[split][class_name] = sorted(
                    rng.sample(readable_files, target),
                    key=lambda file_path: file_path.name.lower(),
                )
            else:
                plan.selected[split][class_name] = readable_files

        duplicates = duplicate_names(split_paths)
        if duplicates:
            plan.source_duplicate_filenames[split] = duplicates

    for split in SPLITS:
        selected_paths = [path for class_paths in plan.selected[split].values() for path in class_paths]
        duplicates = duplicate_names(selected_paths)
        if duplicates:
            plan.output_duplicate_filenames[split] = duplicates

    return plan


def copy_selected_files(plan: CopyPlan, output_root: Path) -> None:
    for split in SPLITS:
        print(f"[copy] {split}")
        for class_name in CLASSES:
            destination_dir = output_root / split / class_name
            selected_files = plan.selected[split][class_name]
            print(f"  {class_name}: {len(selected_files)} files")
            for source_path in selected_files:
                destination_path = destination_dir / source_path.name
                if source_path.resolve() == destination_path.resolve():
                    raise RuntimeError(f"Refusing to copy a file onto itself: {source_path}")
                shutil.copy2(source_path, destination_path)


def count_output(output_root: Path) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {}
    for split in SPLITS:
        counts[split] = Counter()
        for class_name in CLASSES:
            counts[split][class_name] = len(image_files(output_root / split / class_name))
    return counts


def class_percentages(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts[class_name] for class_name in CLASSES)
    if total == 0:
        return {class_name: 0.0 for class_name in CLASSES}
    return {class_name: counts[class_name] * 100.0 / total for class_name in CLASSES}


def imbalance_ratio(counts: Counter[str]) -> float:
    positive = [counts[class_name] for class_name in CLASSES if counts[class_name] > 0]
    if not positive:
        return 0.0
    return max(positive) / min(positive)


def leakage_sets(plan: CopyPlan) -> dict[str, int]:
    file_names = {
        split: {path.name.lower() for class_paths in plan.selected[split].values() for path in class_paths}
        for split in SPLITS
    }
    source_stems = {
        split: {source_image_stem(path).lower() for class_paths in plan.selected[split].values() for path in class_paths}
        for split in SPLITS
    }
    return {
        "filename_training_validation": len(file_names["Training"] & file_names["Validation"]),
        "filename_training_test": len(file_names["Training"] & file_names["Test"]),
        "filename_validation_test": len(file_names["Validation"] & file_names["Test"]),
        "source_stem_training_validation": len(source_stems["Training"] & source_stems["Validation"]),
        "source_stem_training_test": len(source_stems["Training"] & source_stems["Test"]),
        "source_stem_validation_test": len(source_stems["Validation"] & source_stems["Test"]),
    }


def write_counts_csv(output_root: Path, counts: dict[str, Counter[str]]) -> None:
    reports_root = output_root / "reports"
    path = reports_root / "balanced_dataset_counts.csv"
    fieldnames = ["split", "Healthy", "Mild", "Moderate", "Severe", "total", "Healthy_pct", "Mild_pct", "Moderate_pct", "Severe_pct"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        total_counts = Counter()
        for split in SPLITS:
            split_counts = counts[split]
            total_counts.update(split_counts)
            writer.writerow(counts_row(split, split_counts))
        writer.writerow(counts_row("Total", total_counts))


def counts_row(split: str, counts: Counter[str]) -> dict[str, int | str]:
    percentages = class_percentages(counts)
    row: dict[str, int | str] = {
        "split": split,
        "Healthy": counts["Healthy"],
        "Mild": counts["Mild"],
        "Moderate": counts["Moderate"],
        "Severe": counts["Severe"],
        "total": sum(counts[class_name] for class_name in CLASSES),
    }
    for class_name in CLASSES:
        row[f"{class_name}_pct"] = f"{percentages[class_name]:.2f}"
    return row


def write_selected_training_files(output_root: Path, plan: CopyPlan) -> None:
    path = output_root / "reports" / "selected_training_files.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "source_path", "output_path", "source_image_stem"])
        writer.writeheader()
        for class_name in CLASSES:
            for source_path in plan.selected["Training"][class_name]:
                writer.writerow(
                    {
                        "class": class_name,
                        "source_path": str(source_path),
                        "output_path": str(output_root / "Training" / class_name / source_path.name),
                        "source_image_stem": source_image_stem(source_path),
                    }
                )


def write_topup_report(output_root: Path, healthy_count: int, target: int) -> None:
    deficit = max(0, target - healthy_count)
    lines = [
        "Healthy top-up requirement",
        "==========================",
        f"Current Healthy Training ROIs = {healthy_count}",
        f"Target Healthy Training ROIs = {target}",
        f"Additional Healthy ROIs required = {deficit}",
        "",
        "No Healthy images were duplicated.",
        "No synthetic images were generated.",
        "Top-up folders are empty staging folders for future reviewed additions.",
    ]
    (output_root / "reports" / "healthy_topup_requirement.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    source_root: Path,
    output_root: Path,
    counts: dict[str, Counter[str]],
    plan: CopyPlan,
    seed: int,
    target: int,
) -> None:
    reports_root = output_root / "reports"
    training_counts = counts["Training"]
    healthy_deficit = max(0, target - training_counts["Healthy"])
    leakage = leakage_sets(plan)
    total_training = sum(training_counts[class_name] for class_name in CLASSES)

    lines = [
        "Balanced classifier dataset summary",
        "===================================",
        f"Source dataset path: {source_root}",
        f"Balanced dataset path: {output_root}",
        f"Random seed: {seed}",
        f"Training target per non-Healthy majority class: {target}",
        "",
    ]

    for split in SPLITS:
        split_counts = counts[split]
        percentages = class_percentages(split_counts)
        lines.append(f"{split}:")
        for class_name in CLASSES:
            lines.append(f"  {class_name} = {split_counts[class_name]} ({percentages[class_name]:.2f}%)")
        lines.append(f"  Total = {sum(split_counts[class_name] for class_name in CLASSES)}")
        lines.append(f"  Imbalance ratio = {imbalance_ratio(split_counts):.2f}")
        lines.append("")

    lines.extend(
        [
            f"Total training samples = {total_training}",
            f"Healthy deficit required to reach {target} = {healthy_deficit}",
            f"Final target count after Healthy top-up = {target} per class, {target * len(CLASSES)} total training samples",
            "",
            "Healthy top-up:",
            f"  Current Healthy Training ROIs = {training_counts['Healthy']}",
            f"  Target Healthy Training ROIs = {target}",
            f"  Additional Healthy ROIs required = {healthy_deficit}",
            "",
            "Checks:",
            f"  Unreadable images = {len(plan.unreadable_images)}",
            f"  Source duplicate filenames by split = {format_duplicate_report(plan.source_duplicate_filenames)}",
            f"  Selected output duplicate filenames by split = {format_duplicate_report(plan.output_duplicate_filenames)}",
            f"  Filename overlap Training/Validation = {leakage['filename_training_validation']}",
            f"  Filename overlap Training/Test = {leakage['filename_training_test']}",
            f"  Filename overlap Validation/Test = {leakage['filename_validation_test']}",
            f"  Source-stem overlap Training/Validation = {leakage['source_stem_training_validation']}",
            f"  Source-stem overlap Training/Test = {leakage['source_stem_training_test']}",
            f"  Source-stem overlap Validation/Test = {leakage['source_stem_validation_test']}",
            "",
            "Confirmations:",
            "  Original classification_levels files were untouched.",
            "  No synthetic images were created.",
            "  No duplicate oversampling was used.",
            "  Validation and Test were copied exactly, with no downsampling or augmentation.",
        ]
    )

    if plan.unreadable_images:
        lines.append("")
        lines.append("Unreadable images:")
        lines.extend(f"  {path}" for path in plan.unreadable_images)

    (reports_root / "balanced_dataset_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_duplicate_report(duplicates: dict[str, list[str]]) -> str:
    if not duplicates:
        return "none"
    parts = [f"{split}: {len(names)}" for split, names in sorted(duplicates.items())]
    return "; ".join(parts)


def validate_final_counts(counts: dict[str, Counter[str]], target: int) -> None:
    expected_training = {"Healthy": 244, "Mild": target, "Moderate": target, "Severe": target}
    expected_validation = {"Healthy": 30, "Mild": 1080, "Moderate": 755, "Severe": 320}
    expected_test = {"Healthy": 35, "Mild": 1206, "Moderate": 713, "Severe": 221}
    expected = {"Training": expected_training, "Validation": expected_validation, "Test": expected_test}

    errors: list[str] = []
    for split, split_expected in expected.items():
        for class_name, expected_count in split_expected.items():
            actual = counts[split][class_name]
            if actual != expected_count:
                errors.append(f"{split}/{class_name}: expected {expected_count}, found {actual}")

    if errors:
        raise RuntimeError("Final count validation failed:\n" + "\n".join(errors))


def print_planned_actions(source_root: Path, output_root: Path, seed: int, target: int) -> None:
    print("Planned actions and paths")
    print("=========================")
    print(f"Source dataset path: {source_root}")
    print(f"New balanced dataset path: {output_root}")
    print(f"Random seed: {seed}")
    print(f"Training target per class: {target}")
    print("")
    print("Actions:")
    print("  1. Treat classification_levels as read-only.")
    print("  2. Create classification_balanced split/class folders.")
    print("  3. Copy all Training/Healthy files without duplication.")
    print("  4. Randomly select 600 unique Training/Mild files.")
    print("  5. Randomly select 600 unique Training/Moderate files.")
    print("  6. Randomly select 600 unique Training/Severe files.")
    print("  7. Copy Validation and Test exactly as-is.")
    print("  8. Create empty healthy_topup/real, synthetic, approved, rejected folders.")
    print("  9. Write reports and validate counts/leakage.")


def print_final_output(source_root: Path, output_root: Path, counts: dict[str, Counter[str]], target: int) -> None:
    healthy_deficit = max(0, target - counts["Training"]["Healthy"])
    print("")
    print("Final output")
    print("============")
    print(f"1. Source dataset path: {source_root}")
    print(f"2. New balanced dataset path: {output_root}")
    for number, split in enumerate(SPLITS, start=3):
        split_counts = counts[split]
        print(
            f"{number}. {split} counts: Healthy={split_counts['Healthy']}, Mild={split_counts['Mild']}, "
            f"Moderate={split_counts['Moderate']}, Severe={split_counts['Severe']}"
        )
    print(f"6. Healthy deficit: {healthy_deficit}")
    print("7. Original classification_levels files were untouched.")
    print("8. No synthetic images were created.")
    print("9. No duplicate oversampling was used.")
    print("10. Validation/Test were preserved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a balanced gingivitis classification dataset.")
    parser.add_argument("--project-root", type=Path, default=project_root(), help="Project root.")
    parser.add_argument("--source-root", type=Path, default=None, help="Read-only classification_levels folder.")
    parser.add_argument("--output-root", type=Path, default=None, help="Generated classification_balanced folder.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Fixed random seed.")
    parser.add_argument("--target", type=int, default=TRAINING_TARGET_PER_CLASS, help="Training target per class.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without copying files.")
    args = parser.parse_args()

    root = args.project_root.resolve()
    source_root = args.source_root.resolve() if args.source_root else default_source_root(root).resolve()
    output_root = args.output_root.resolve() if args.output_root else default_output_root(root).resolve()

    print_planned_actions(source_root, output_root, args.seed, args.target)
    if args.dry_run:
        return

    ensure_output_structure(output_root)
    plan = build_copy_plan(source_root, args.seed, args.target)

    if plan.unreadable_images:
        raise RuntimeError(f"Found {len(plan.unreadable_images)} unreadable source images. See console output.")
    if plan.output_duplicate_filenames:
        raise RuntimeError(f"Selected output contains duplicate filenames: {plan.output_duplicate_filenames}")

    copy_selected_files(plan, output_root)
    counts = count_output(output_root)
    validate_final_counts(counts, args.target)
    write_counts_csv(output_root, counts)
    write_selected_training_files(output_root, plan)
    write_topup_report(output_root, counts["Training"]["Healthy"], args.target)
    write_summary(source_root, output_root, counts, plan, args.seed, args.target)
    print_final_output(source_root, output_root, counts, args.target)


if __name__ == "__main__":
    main()
