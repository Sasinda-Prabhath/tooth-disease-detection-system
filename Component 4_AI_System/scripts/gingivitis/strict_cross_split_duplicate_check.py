"""Strict cross-split duplicate verification for gingivitis ROI pairs.

Reads the final audit near-duplicate warnings, recalculates exact hashes,
dHash, pHash, and SSIM, then classifies each warning without modifying any
dataset image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError as exc:  # pragma: no cover - user-facing dependency guard.
    raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc


SPLITS = ("Training", "Validation", "Test")
CLASSES = ("Healthy", "Mild", "Moderate", "Severe")
PHASH_SIZE = 32
PHASH_LOW_SIZE = 8
HASH_RESIZE = Image.Resampling.LANCZOS
SSIM_SIZE = 256


@dataclass(frozen=True)
class ImageMetrics:
    rel_path: str
    abs_path: Path
    split: str
    class_name: str
    filename: str
    source_stem: str
    sha256: str
    dhash: int
    phash: int


@dataclass
class PairResult:
    category: str
    file_a: str
    split_a: str
    class_a: str
    source_stem_a: str
    file_b: str
    split_b: str
    class_b: str
    source_stem_b: str
    exact_hash_match: bool
    same_source_stem: bool
    dhash_distance: int
    phash_distance: int
    ssim: float | None
    reason: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced"


def default_reports_dir(root: Path) -> Path:
    return default_dataset_dir(root) / "reports"


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixels(image: Image.Image) -> list[int]:
    get_pixels = getattr(image, "get_flattened_data", image.getdata)
    return list(get_pixels())


def dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), HASH_RESIZE)
    values = pixels(gray)
    result = 0
    bit = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            if values[offset + col] > values[offset + col + 1]:
                result |= 1 << bit
            bit += 1
    return result


def dct_basis(size: int) -> list[list[float]]:
    basis: list[list[float]] = []
    for u in range(PHASH_LOW_SIZE):
        row = []
        alpha = math.sqrt(1 / size) if u == 0 else math.sqrt(2 / size)
        for x in range(size):
            row.append(alpha * math.cos(((2 * x + 1) * u * math.pi) / (2 * size)))
        basis.append(row)
    return basis


PHASH_BASIS = dct_basis(PHASH_SIZE)


def phash(image: Image.Image) -> int:
    gray = image.convert("L").resize((PHASH_SIZE, PHASH_SIZE), HASH_RESIZE)
    values = pixels(gray)
    matrix = [
        [float(values[row * PHASH_SIZE + col]) for col in range(PHASH_SIZE)]
        for row in range(PHASH_SIZE)
    ]
    coefficients: list[float] = []
    for u in range(PHASH_LOW_SIZE):
        for v in range(PHASH_LOW_SIZE):
            total = 0.0
            basis_u = PHASH_BASIS[u]
            basis_v = PHASH_BASIS[v]
            for x in range(PHASH_SIZE):
                row = matrix[x]
                bx = basis_u[x]
                for y in range(PHASH_SIZE):
                    total += row[y] * bx * basis_v[y]
            coefficients.append(total)

    comparable = coefficients[1:]
    median = sorted(comparable)[len(comparable) // 2]
    result = 0
    for index, coefficient in enumerate(coefficients):
        if coefficient > median:
            result |= 1 << index
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def ssim_score(path_a: Path, path_b: Path) -> float:
    with Image.open(path_a) as image_a, Image.open(path_b) as image_b:
        gray_a = ImageOps.exif_transpose(image_a).convert("L").resize((SSIM_SIZE, SSIM_SIZE), HASH_RESIZE)
        gray_b = ImageOps.exif_transpose(image_b).convert("L").resize((SSIM_SIZE, SSIM_SIZE), HASH_RESIZE)

    values_a = [float(value) for value in pixels(gray_a)]
    values_b = [float(value) for value in pixels(gray_b)]
    count = len(values_a)
    mean_a = sum(values_a) / count
    mean_b = sum(values_b) / count
    var_a = sum((value - mean_a) ** 2 for value in values_a) / (count - 1)
    var_b = sum((value - mean_b) ** 2 for value in values_b) / (count - 1)
    cov = sum((left - mean_a) * (right - mean_b) for left, right in zip(values_a, values_b)) / (count - 1)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    denominator = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    if denominator == 0:
        return 1.0 if values_a == values_b else 0.0
    return ((2 * mean_a * mean_b + c1) * (2 * cov + c2)) / denominator


def collect_dataset_metrics(dataset_dir: Path) -> dict[str, ImageMetrics]:
    metrics: dict[str, ImageMetrics] = {}
    for split in SPLITS:
        for class_name in CLASSES:
            class_dir = dataset_dir / split / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir(), key=lambda item: item.name.lower()):
                if not path.is_file():
                    continue
                rel_path = path.relative_to(dataset_dir).as_posix()
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    metrics[rel_path] = ImageMetrics(
                        rel_path=rel_path,
                        abs_path=path,
                        split=split,
                        class_name=class_name,
                        filename=path.name,
                        source_stem=source_stem_from_filename(path.name),
                        sha256=file_sha256(path),
                        dhash=dhash(image),
                        phash=phash(image),
                    )
    return metrics


def read_warning_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            file_a = row.get("file_a", "")
            file_b = row.get("file_b", "")
            if file_a and file_b:
                pairs.append((file_a, file_b))
    return pairs


def classify_pair(left: ImageMetrics, right: ImageMetrics) -> PairResult:
    exact = left.sha256 == right.sha256
    same_source = left.source_stem == right.source_stem
    d_distance = hamming_distance(left.dhash, right.dhash)
    p_distance = hamming_distance(left.phash, right.phash)

    needs_ssim = exact or d_distance <= 1 or p_distance <= 1 or same_source
    ssim = ssim_score(left.abs_path, right.abs_path) if needs_ssim else None

    if exact:
        category = "likely true duplicate"
        reason = "same SHA-256 hash across splits"
    elif same_source and d_distance <= 1 and p_distance <= 1 and ssim is not None and ssim >= 0.90:
        category = "likely true duplicate"
        reason = "same source stem with dHash<=1, pHash<=1, and high SSIM"
    elif d_distance <= 1 and p_distance <= 1 and ssim is not None and ssim >= 0.88:
        category = "very strong near-duplicate"
        reason = "dHash<=1, pHash<=1, and SSIM>=0.88"
    elif p_distance <= 1 and ssim is not None and ssim >= 0.92:
        category = "very strong near-duplicate"
        reason = "pHash<=1 with high SSIM"
    elif d_distance <= 1 and ssim is not None and ssim >= 0.95:
        category = "very strong near-duplicate"
        reason = "dHash<=1 with very high SSIM"
    else:
        category = "visually similar dental ROI only"
        if p_distance <= 1 or d_distance <= 1:
            reason = "strict hash threshold met but SSIM/source evidence is not strong enough"
        else:
            reason = "low dHash audit warning did not persist under stricter pHash/SSIM filtering"

    return PairResult(
        category=category,
        file_a=left.rel_path,
        split_a=left.split,
        class_a=left.class_name,
        source_stem_a=left.source_stem,
        file_b=right.rel_path,
        split_b=right.split,
        class_b=right.class_name,
        source_stem_b=right.source_stem,
        exact_hash_match=exact,
        same_source_stem=same_source,
        dhash_distance=d_distance,
        phash_distance=p_distance,
        ssim=ssim,
        reason=reason,
    )


def write_results_csv(path: Path, results: list[PairResult]) -> None:
    fieldnames = [
        "category",
        "file_a",
        "split_a",
        "class_a",
        "source_stem_a",
        "file_b",
        "split_b",
        "class_b",
        "source_stem_b",
        "same_source_stem",
        "exact_hash_match",
        "dhash_distance",
        "phash_distance",
        "ssim",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "category": result.category,
                    "file_a": result.file_a,
                    "split_a": result.split_a,
                    "class_a": result.class_a,
                    "source_stem_a": result.source_stem_a,
                    "file_b": result.file_b,
                    "split_b": result.split_b,
                    "class_b": result.class_b,
                    "source_stem_b": result.source_stem_b,
                    "same_source_stem": result.same_source_stem,
                    "exact_hash_match": result.exact_hash_match,
                    "dhash_distance": result.dhash_distance,
                    "phash_distance": result.phash_distance,
                    "ssim": "" if result.ssim is None else f"{result.ssim:.6f}",
                    "reason": result.reason,
                }
            )


def write_summary(path: Path, total_warnings: int, results: list[PairResult], status: str) -> None:
    category_counts = {}
    for category in ("likely true duplicate", "very strong near-duplicate", "visually similar dental ROI only"):
        category_counts[category] = sum(1 for result in results if result.category == category)

    strict_dhash = sum(1 for result in results if result.dhash_distance <= 1)
    strict_phash = sum(1 for result in results if result.phash_distance <= 1)
    both_strict = sum(1 for result in results if result.dhash_distance <= 1 and result.phash_distance <= 1)
    same_source = sum(1 for result in results if result.same_source_stem)
    exact = sum(1 for result in results if result.exact_hash_match)
    ssim_count = sum(1 for result in results if result.ssim is not None)
    suspicious = category_counts["likely true duplicate"] + category_counts["very strong near-duplicate"]

    lines = [
        "Strict cross-split duplicate check summary",
        "==========================================",
        f"input cross-split near-duplicate warnings reviewed: {total_warnings}",
        f"exact hash matches across reviewed pairs: {exact}",
        f"pHash distance <= 1 pairs: {strict_phash}",
        f"dHash distance <= 1 pairs: {strict_dhash}",
        f"both pHash <= 1 and dHash <= 1 pairs: {both_strict}",
        f"SSIM calculated for pairs: {ssim_count}",
        "",
        "Strict categories:",
        f"- likely true duplicate: {category_counts['likely true duplicate']}",
        f"- very strong near-duplicate: {category_counts['very strong near-duplicate']}",
        f"- visually similar dental ROI only: {category_counts['visually similar dental ROI only']}",
        "",
        f"suspicious pairs remaining after stricter filtering: {suspicious}",
        f"suspicious pairs from same source image: {sum(1 for result in results if result.same_source_stem and result.category != 'visually similar dental ROI only')}",
        f"all reviewed pairs from same source image: {same_source}",
        "",
        f"FINAL STATUS: {status}",
        "",
        "No Training, Validation, or Test images were modified, moved, renamed, deleted, resized, or overwritten.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = project_root()
    reports_dir = default_reports_dir(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(default_dataset_dir(root)))
    parser.add_argument("--input-near-duplicates", default=str(reports_dir / "final_dataset_near_duplicates.csv"))
    parser.add_argument("--output-csv", default=str(reports_dir / "strict_cross_split_duplicate_check.csv"))
    parser.add_argument("--output-summary", default=str(reports_dir / "strict_cross_split_duplicate_summary.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    warning_pairs = read_warning_pairs(Path(args.input_near_duplicates))
    metrics = collect_dataset_metrics(dataset_dir)

    results: list[PairResult] = []
    for file_a, file_b in warning_pairs:
        left = metrics[file_a]
        right = metrics[file_b]
        results.append(classify_pair(left, right))

    suspicious = sum(1 for result in results if result.category in {"likely true duplicate", "very strong near-duplicate"})
    same_source_suspicious = sum(
        1
        for result in results
        if result.same_source_stem and result.category in {"likely true duplicate", "very strong near-duplicate"}
    )
    status = "REVIEW REQUIRED" if suspicious or same_source_suspicious else "SAFE TO TRAIN"

    write_results_csv(Path(args.output_csv), results)
    write_summary(Path(args.output_summary), len(warning_pairs), results, status)

    print("Strict cross-split duplicate check")
    print("==================================")
    print(f"warnings reviewed: {len(warning_pairs)}")
    print(f"likely true duplicate: {sum(1 for result in results if result.category == 'likely true duplicate')}")
    print(f"very strong near-duplicate: {sum(1 for result in results if result.category == 'very strong near-duplicate')}")
    print(f"visually similar dental ROI only: {sum(1 for result in results if result.category == 'visually similar dental ROI only')}")
    print(f"suspicious pairs remaining: {suspicious}")
    print(f"suspicious same-source pairs: {same_source_suspicious}")
    print(f"FINAL STATUS: {status}")


if __name__ == "__main__":
    main()
