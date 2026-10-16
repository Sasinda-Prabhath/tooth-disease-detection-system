"""Analyze class balance for the generated gingivitis classification dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


SPLITS = ("Training", "Validation", "Test")
CLASS_NAMES = ("Healthy", "Mild", "Moderate", "Severe")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_classification_root(source_root: Path) -> Path:
    return source_root / "datasets" / "gingivitis" / "classification_levels"


def count_class_samples(classification_root: Path) -> dict[str, Counter[str]]:
    counts_by_split: dict[str, Counter[str]] = {}
    for split in SPLITS:
        split_counts: Counter[str] = Counter()
        for class_name in CLASS_NAMES:
            class_dir = classification_root / split / class_name
            if class_dir.exists():
                split_counts[class_name] = sum(
                    1
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
            else:
                split_counts[class_name] = 0
        counts_by_split[split] = split_counts
    return counts_by_split


def calculate_percentages(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts[class_name] for class_name in CLASS_NAMES)
    if total == 0:
        return {class_name: 0.0 for class_name in CLASS_NAMES}
    return {class_name: counts[class_name] * 100.0 / total for class_name in CLASS_NAMES}


def imbalance_ratio(counts: Counter[str]) -> float:
    positive_counts = [counts[class_name] for class_name in CLASS_NAMES if counts[class_name] > 0]
    if not positive_counts:
        return 0.0
    minimum = min(positive_counts)
    maximum = max(positive_counts)
    return maximum / minimum


def underrepresented_classes(counts: Counter[str]) -> list[str]:
    total = sum(counts[class_name] for class_name in CLASS_NAMES)
    if total == 0:
        return list(CLASS_NAMES)

    average = total / len(CLASS_NAMES)
    maximum = max(counts[class_name] for class_name in CLASS_NAMES)
    threshold = min(average, maximum * 0.5)
    return [class_name for class_name in CLASS_NAMES if counts[class_name] < threshold]


def print_balance_report(classification_root: Path, counts_by_split: dict[str, Counter[str]]) -> None:
    print("Classification dataset balance analysis")
    print("=======================================")
    print(f"Classification root: {classification_root}")

    total_counts: Counter[str] = Counter()
    for split in SPLITS:
        counts = counts_by_split[split]
        total_counts.update(counts)
        percentages = calculate_percentages(counts)
        ratio = imbalance_ratio(counts)
        underrepresented = underrepresented_classes(counts)
        print("")
        print(f"{split}:")
        for class_name in CLASS_NAMES:
            print(f"  {class_name}: {counts[class_name]} ({percentages[class_name]:.2f}%)")
        print(f"  Imbalance ratio: {ratio:.2f}")
        print(f"  Underrepresented classes: {', '.join(underrepresented) if underrepresented else 'none'}")

    total_percentages = calculate_percentages(total_counts)
    total_ratio = imbalance_ratio(total_counts)
    total_underrepresented = underrepresented_classes(total_counts)
    healthy_is_underrepresented = "Healthy" in total_underrepresented

    print("")
    print("Total:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {total_counts[class_name]} ({total_percentages[class_name]:.2f}%)")
    print(f"  Imbalance ratio: {total_ratio:.2f}")
    print(f"  Underrepresented classes: {', '.join(total_underrepresented) if total_underrepresented else 'none'}")
    print(f"  Healthy underrepresented: {'YES' if healthy_is_underrepresented else 'NO'}")
    print("")
    print("No oversampling or synthetic image generation was performed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated gingivitis classification class balance.")
    parser.add_argument("--source-root", type=Path, default=project_root(), help="Project/source dataset root.")
    parser.add_argument("--classification-root", type=Path, default=None, help="Generated classification_levels folder.")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    classification_root = args.classification_root.resolve() if args.classification_root else default_classification_root(source_root)
    counts_by_split = count_class_samples(classification_root)
    print_balance_report(classification_root, counts_by_split)


if __name__ == "__main__":
    main()
