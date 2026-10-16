"""Select approved synthetic Healthy gingival ROI candidates.

The script reviews every image in synthetic_rois with deterministic image
quality metrics and source-balanced selection. It copies selected images to
healthy_topup/approved and the remaining images to healthy_topup/rejected only
when --execute is supplied. By default it performs a dry run and writes no
files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageFilter, ImageOps, ImageStat
except ImportError as exc:  # pragma: no cover - user-facing dependency guard.
    raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TARGET_APPROVED = 356
MAX_PER_SOURCE = 3
SEED = 42
MIN_CROP_SIDE = 180
MIN_GOOD_SCORE = 48.0
NEAR_DUPLICATE_DISTANCE = 6


@dataclass
class RoiReview:
    path: Path
    filename: str
    source_image: str
    readable: bool = False
    width: int = 0
    height: int = 0
    mean: float = 0.0
    stddev: float = 0.0
    entropy: float = 0.0
    sharpness_score: float = 0.0
    tooth_fraction: float = 0.0
    gingiva_fraction: float = 0.0
    dark_fraction: float = 0.0
    bright_fraction: float = 0.0
    exact_hash: str = ""
    dhash: int | None = None
    duplicate_status: str = "unique"
    score: float = 0.0
    quality_status: str = "pending"
    reason: str = ""
    selected: bool = False
    selected_from_source_count: int = 0
    source_roi_count: int = 0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_candidate_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced" / "healthy_topup" / "synthetic_rois"


def default_approved_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced" / "healthy_topup" / "approved"


def default_rejected_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced" / "healthy_topup" / "rejected"


def default_topup_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced" / "healthy_topup"


def default_classification_reports_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced" / "reports"


def default_reports_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "reports"


def source_from_roi_name(filename: str) -> str:
    stem = Path(filename).stem
    marker = "_roi_"
    if marker in stem:
        return stem.split(marker, 1)[0] + ".jpg"
    return "UNKNOWN"


def find_images(candidate_dir: Path) -> list[Path]:
    if not candidate_dir.exists():
        return []
    return sorted(
        [path for path in candidate_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entropy(gray: Image.Image) -> float:
    histogram = gray.histogram()
    total = sum(histogram)
    if total == 0:
        return 0.0
    value = 0.0
    for count in histogram:
        if count:
            probability = count / total
            value -= probability * math.log2(probability)
    return value


def color_fractions(image: Image.Image) -> tuple[float, float, float, float]:
    sample = image.convert("RGB")
    sample.thumbnail((160, 160), Image.Resampling.LANCZOS)
    get_pixels = getattr(sample, "get_flattened_data", sample.getdata)
    pixels = list(get_pixels())
    total = len(pixels)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    tooth = 0
    gingiva = 0
    dark = 0
    bright = 0
    for red, green, blue in pixels:
        maximum = max(red, green, blue)
        minimum = min(red, green, blue)
        saturation = maximum - minimum
        mean = (red + green + blue) / 3

        if mean < 35:
            dark += 1
        if mean > 235:
            bright += 1
        if maximum > 145 and saturation < 78 and mean > 132:
            tooth += 1
        if red > 62 and red > green * 1.05 and red > blue * 1.05 and saturation > 16:
            gingiva += 1

    return tooth / total, gingiva / total, dark / total, bright / total


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


def edge_sharpness(gray: Image.Image) -> float:
    sample = gray.resize((256, 256), Image.Resampling.LANCZOS)
    edges = sample.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    return float(stat.mean[0] + stat.stddev[0])


def load_report_duplicate_flags(reports_dir: Path) -> dict[str, str]:
    duplicate_path = reports_dir / "synthetic_roi_duplicates.csv"
    statuses: dict[str, str] = {}
    if not duplicate_path.exists():
        return statuses

    with duplicate_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            duplicate_type = row.get("duplicate_type", "")
            item = row.get("item", "")
            if not item:
                continue
            if duplicate_type == "crop_exact_duplicate":
                statuses[item] = "exact_duplicate_reported"
            elif duplicate_type == "crop_near_duplicate_warning" and item not in statuses:
                statuses[item] = "near_duplicate_reported"
    return statuses


def review_image(path: Path) -> RoiReview:
    review = RoiReview(path=path, filename=path.name, source_image=source_from_roi_name(path.name))
    try:
        review.exact_hash = file_sha256(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            review.width, review.height = image.size
            gray = image.convert("L")
            stat = ImageStat.Stat(gray)
            review.mean = float(stat.mean[0])
            review.stddev = float(stat.stddev[0])
            review.entropy = entropy(gray)
            review.sharpness_score = edge_sharpness(gray)
            (
                review.tooth_fraction,
                review.gingiva_fraction,
                review.dark_fraction,
                review.bright_fraction,
            ) = color_fractions(image)
            review.dhash = dhash(image)
        review.readable = True
    except Exception as exc:  # pragma: no cover - defensive for damaged files.
        review.quality_status = "rejected"
        review.reason = f"unreadable image: {exc}"
        return review

    reject_reasons: list[str] = []
    if min(review.width, review.height) < MIN_CROP_SIDE:
        reject_reasons.append("crop too small")
    if review.mean < 32:
        reject_reasons.append("too dark")
    if review.mean > 242:
        reject_reasons.append("too bright")
    if review.stddev < 14 or review.entropy < 3.0:
        reject_reasons.append("low texture/uniform")
    if review.sharpness_score < 15:
        reject_reasons.append("low sharpness")
    if review.dark_fraction > 0.52:
        reject_reasons.append("too much dark mouth cavity")
    if review.bright_fraction > 0.70:
        reject_reasons.append("too much bright tooth/glare")
    if review.tooth_fraction < 0.012:
        reject_reasons.append("too little tooth reference")
    if review.gingiva_fraction < 0.022:
        reject_reasons.append("too little gingiva")

    review.score = quality_score(review)
    if reject_reasons:
        review.quality_status = "rejected"
        review.reason = "; ".join(reject_reasons)
    elif review.score < MIN_GOOD_SCORE:
        review.quality_status = "borderline"
        review.reason = f"quality score below preferred threshold ({review.score:.1f})"
    else:
        review.quality_status = "good"
        review.reason = "clear tooth and gingiva context with acceptable sharpness/exposure"
    return review


def quality_score(review: RoiReview) -> float:
    tooth_score = min(review.tooth_fraction / 0.16, 1.0) * 24
    gingiva_score = min(review.gingiva_fraction / 0.18, 1.0) * 30
    sharp_score = min(review.sharpness_score / 45.0, 1.0) * 20
    texture_score = min(review.entropy / 7.0, 1.0) * 14
    contrast_score = min(review.stddev / 62.0, 1.0) * 12
    dark_penalty = max(0.0, review.dark_fraction - 0.24) * 30
    bright_penalty = max(0.0, review.bright_fraction - 0.42) * 18
    balance_penalty = abs(review.tooth_fraction - review.gingiva_fraction) * 8
    return tooth_score + gingiva_score + sharp_score + texture_score + contrast_score - dark_penalty - bright_penalty - balance_penalty


def mark_detected_duplicates(reviews: list[RoiReview], report_flags: dict[str, str]) -> None:
    seen_hashes: dict[str, str] = {}
    seen_dhashes: list[tuple[int, str]] = []

    for review in sorted(reviews, key=lambda item: item.filename.lower()):
        report_flag = report_flags.get(review.filename)
        if report_flag:
            review.duplicate_status = report_flag

        if review.exact_hash:
            duplicate_of = seen_hashes.get(review.exact_hash)
            if duplicate_of:
                review.duplicate_status = f"exact_duplicate_of:{duplicate_of}"
            else:
                seen_hashes[review.exact_hash] = review.filename

        if review.dhash is None or review.duplicate_status.startswith("exact_duplicate"):
            continue

        nearest_name = ""
        nearest_distance = 65
        for previous_hash, previous_name in seen_dhashes:
            distance = hamming_distance(review.dhash, previous_hash)
            if distance < nearest_distance:
                nearest_name = previous_name
                nearest_distance = distance
        if nearest_distance <= NEAR_DUPLICATE_DISTANCE and review.duplicate_status == "unique":
            review.duplicate_status = f"near_duplicate_of:{nearest_name}:distance_{nearest_distance}"
        seen_dhashes.append((review.dhash, review.filename))


def select_reviews(reviews: list[RoiReview], target: int, max_per_source: int, seed: int) -> None:
    rng = random.Random(seed)
    by_source: dict[str, list[RoiReview]] = defaultdict(list)
    for review in reviews:
        by_source[review.source_image].append(review)

    for source_reviews in by_source.values():
        for review in source_reviews:
            review.source_roi_count = len(source_reviews)

    eligible = [
        review
        for review in reviews
        if review.readable
        and review.quality_status in {"good", "borderline"}
        and not review.duplicate_status.startswith("exact_duplicate")
    ]
    good = [review for review in eligible if review.quality_status == "good"]
    pool = good if len(good) >= target else eligible

    for source_reviews in by_source.values():
        source_reviews.sort(key=lambda item: (item.score, rng.random()), reverse=True)

    selected: list[RoiReview] = []
    selected_names: set[str] = set()
    selected_by_source: Counter[str] = Counter()

    near_duplicate_selected_sources: set[str] = set()

    def try_add(review: RoiReview, allow_near_duplicate: bool) -> bool:
        if review.filename in selected_names:
            return False
        if selected_by_source[review.source_image] >= max_per_source:
            return False
        if review not in pool:
            return False
        if review.duplicate_status.startswith("near_duplicate") or review.duplicate_status == "near_duplicate_reported":
            if not allow_near_duplicate:
                return False
            if review.source_image in near_duplicate_selected_sources:
                return False
        selected.append(review)
        selected_names.add(review.filename)
        selected_by_source[review.source_image] += 1
        if review.duplicate_status.startswith("near_duplicate") or review.duplicate_status == "near_duplicate_reported":
            near_duplicate_selected_sources.add(review.source_image)
        return True

    # One high-quality ROI per source first, then second and third passes. This
    # keeps source diversity high while still reaching the exact target.
    sources = sorted(by_source, key=lambda name: (len(by_source[name]), name))
    for pass_index in range(max_per_source):
        ordered_sources = sources[:]
        rng.shuffle(ordered_sources)
        for source in ordered_sources:
            if len(selected) >= target:
                break
            if selected_by_source[source] != pass_index:
                continue
            for review in by_source[source]:
                if try_add(review, allow_near_duplicate=False):
                    break
        if len(selected) >= target:
            break

    if len(selected) < target:
        fallback = sorted(pool, key=lambda item: (item.score, rng.random()), reverse=True)
        for review in fallback:
            if len(selected) >= target:
                break
            try_add(review, allow_near_duplicate=True)

    for review in selected:
        review.selected = True
    final_counts = Counter(review.source_image for review in selected)
    for review in reviews:
        review.selected_from_source_count = final_counts[review.source_image]
        if review.selected:
            if review.duplicate_status == "unique":
                review.reason = "selected: high quality, source-balanced ROI"
            else:
                review.reason = f"selected: high quality/source-balanced; {review.duplicate_status}"
        elif review.quality_status == "good" and len(selected) >= target:
            review.reason = "not selected: target reached with more diverse/higher-ranked candidates"
        elif review.quality_status == "borderline" and len(good) >= target:
            review.reason = "not selected: borderline quality and enough stronger candidates were available"


def copy_review_files(reviews: list[RoiReview], approved_dir: Path, rejected_dir: Path) -> None:
    approved_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    for review in reviews:
        destination_dir = approved_dir if review.selected else rejected_dir
        destination = destination_dir / review.filename
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
        shutil.copy2(review.path, destination)


def write_manifests(reviews: list[RoiReview], topup_dir: Path) -> None:
    approved_path = topup_dir / "approved_manifest.csv"
    rejected_path = topup_dir / "rejected_manifest.csv"

    with approved_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "source_image", "reason"])
        writer.writeheader()
        for review in sorted([item for item in reviews if item.selected], key=lambda item: item.filename):
            writer.writerow({"filename": review.filename, "source_image": review.source_image, "reason": review.reason})

    with rejected_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "source_image", "reason"])
        writer.writeheader()
        for review in sorted([item for item in reviews if not item.selected], key=lambda item: item.filename):
            writer.writerow({"filename": review.filename, "source_image": review.source_image, "reason": review.reason})


def write_selection_csv(reviews: list[RoiReview], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "approved_healthy_selection.csv"
    fieldnames = [
        "source_image",
        "roi_filename",
        "selected",
        "selection_reason",
        "sharpness_score",
        "duplicate_status",
        "source_roi_count",
        "selected_from_source_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for review in sorted(reviews, key=lambda item: item.filename):
            writer.writerow(
                {
                    "source_image": review.source_image,
                    "roi_filename": review.filename,
                    "selected": review.selected,
                    "selection_reason": review.reason,
                    "sharpness_score": f"{review.sharpness_score:.3f}",
                    "duplicate_status": review.duplicate_status,
                    "source_roi_count": review.source_roi_count,
                    "selected_from_source_count": review.selected_from_source_count,
                }
            )


def write_summary(reviews: list[RoiReview], reports_dir: Path, target: int) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    selected = [review for review in reviews if review.selected]
    rejected = [review for review in reviews if not review.selected]
    represented_sources = {review.source_image for review in selected}
    duplicate_counts = Counter(
        "exact" if review.duplicate_status.startswith("exact_duplicate") else
        "near" if review.duplicate_status.startswith("near_duplicate") or review.duplicate_status == "near_duplicate_reported" else
        "unique"
        for review in reviews
    )
    rejection_reasons = Counter()
    for review in rejected:
        primary = review.reason.split(";", 1)[0]
        rejection_reasons[primary] += 1

    average = len(selected) / len(represented_sources) if represented_sources else 0.0
    lines = [
        "Synthetic Healthy ROI selection summary",
        "=======================================",
        f"total source ROI images: {len(reviews)}",
        f"target approved: {target}",
        f"total approved: {len(selected)}",
        f"total rejected: {len(rejected)}",
        f"unique source images represented: {len(represented_sources)}",
        f"average selected ROIs per source image: {average:.2f}",
        f"exact duplicates detected/excluded: {duplicate_counts['exact']}",
        f"near-duplicates flagged: {duplicate_counts['near']}",
        f"near-duplicates selected only if needed: {sum(1 for review in selected if review.duplicate_status != 'unique')}",
        f"final approved count: {len(selected)}",
        "",
        "Common rejection/not-selected reasons:",
    ]
    for reason, count in rejection_reasons.most_common(12):
        lines.append(f"- {reason}: {count}")
    lines.extend(
        [
            "",
            "Files were copied only. Original synthetic_rois images were not modified, moved, or deleted.",
            "These are candidate Healthy ROI selections for manual dataset review; no medical validation is implied.",
        ]
    )
    (reports_dir / "synthetic_selection_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def review_all(args: argparse.Namespace) -> list[RoiReview]:
    candidate_dir = Path(args.candidate_dir)
    report_flags = load_report_duplicate_flags(Path(args.classification_reports_dir))
    reviews = [review_image(path) for path in find_images(candidate_dir)]
    mark_detected_duplicates(reviews, report_flags)
    select_reviews(reviews, args.target, args.max_per_source, args.seed)
    return reviews


def print_counts(reviews: list[RoiReview], args: argparse.Namespace) -> None:
    selected = [review for review in reviews if review.selected]
    rejected = [review for review in reviews if not review.selected]
    duplicate_counts = Counter(
        "exact" if review.duplicate_status.startswith("exact_duplicate") else
        "near" if review.duplicate_status.startswith("near_duplicate") or review.duplicate_status == "near_duplicate_reported" else
        "unique"
        for review in reviews
    )
    represented_sources = {review.source_image for review in selected}
    print("Synthetic Healthy ROI selection dry run" if not args.execute else "Synthetic Healthy ROI selection")
    print("=======================================")
    print(f"candidate folder: {Path(args.candidate_dir)}")
    print(f"approved folder: {Path(args.approved_dir)}")
    print(f"rejected folder: {Path(args.rejected_dir)}")
    print(f"total source ROI images: {len(reviews)}")
    print(f"target approved: {args.target}")
    print(f"selected/approved count: {len(selected)}")
    print(f"rejected/not-selected count: {len(rejected)}")
    print(f"unique source images represented: {len(represented_sources)}")
    print(f"average selected ROIs per source image: {len(selected) / len(represented_sources):.2f}" if represented_sources else "average selected ROIs per source image: 0.00")
    print(f"exact duplicates excluded: {duplicate_counts['exact']}")
    print(f"near-duplicates flagged: {duplicate_counts['near']}")
    print(f"near-duplicates selected: {sum(1 for review in selected if review.duplicate_status != 'unique')}")
    print(f"max selected from one source: {max(Counter(review.source_image for review in selected).values(), default=0)}")
    print("copy action: " + ("performed" if args.execute else "not performed; re-run with --execute to copy files"))


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", default=str(default_candidate_dir(root)))
    parser.add_argument("--approved-dir", default=str(default_approved_dir(root)))
    parser.add_argument("--rejected-dir", default=str(default_rejected_dir(root)))
    parser.add_argument("--topup-dir", default=str(default_topup_dir(root)))
    parser.add_argument("--classification-reports-dir", default=str(default_classification_reports_dir(root)))
    parser.add_argument("--reports-dir", default=str(default_reports_dir(root)))
    parser.add_argument("--target", type=int, default=TARGET_APPROVED)
    parser.add_argument("--max-per-source", type=int, default=MAX_PER_SOURCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--execute", action="store_true", help="copy approved/rejected files and write manifests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reviews = review_all(args)
    print_counts(reviews, args)
    if not args.execute:
        return

    approved_dir = Path(args.approved_dir)
    rejected_dir = Path(args.rejected_dir)
    existing_approved = list(approved_dir.glob("*")) if approved_dir.exists() else []
    existing_rejected = list(rejected_dir.glob("*")) if rejected_dir.exists() else []
    if existing_approved or existing_rejected:
        raise SystemExit("Destination folders are not empty; refusing to copy without manual cleanup.")

    copy_review_files(reviews, approved_dir, rejected_dir)
    write_manifests(reviews, Path(args.topup_dir))
    write_selection_csv(reviews, Path(args.classification_reports_dir))
    write_summary(reviews, Path(args.reports_dir), args.target)


if __name__ == "__main__":
    main()
