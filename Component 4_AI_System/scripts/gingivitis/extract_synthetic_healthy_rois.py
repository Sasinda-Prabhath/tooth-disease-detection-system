"""Extract candidate Healthy gingival ROI crops from synthetic source images.

This script is intentionally conservative and does not create labels, draw
boxes, or write directly into the classifier Training/Healthy folder. By
default it only prints a preflight plan. Use --execute to write candidate crops
after manual confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageOps, ImageStat
except ImportError as exc:  # pragma: no cover - user-facing dependency guard.
    raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
EXPECTED_SOURCE_COUNT = 150
DEFAULT_MIN_SOURCE_SIDE = 512
DEFAULT_MIN_CROP_SIDE = 180
DEFAULT_MAX_ROIS_PER_SOURCE = 4
DEFAULT_NEAR_DUPLICATE_DISTANCE = 6


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class SourceInfo:
    path: Path
    readable: bool
    width: int = 0
    height: int = 0
    sha256: str = ""
    exact_duplicate: bool = False
    duplicate_of: str = ""
    low_resolution: bool = False
    error: str = ""


@dataclass
class CropAssessment:
    status: str
    score: float
    reason: str = ""
    exact_duplicate: bool = False
    near_duplicate_warning: bool = False
    near_duplicate_of: str = ""
    near_duplicate_distance: int | None = None
    sha256: str = ""
    dhash: int | None = None


@dataclass
class CropRecord:
    source_image: str
    roi_filename: str
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    crop_width: int
    crop_height: int
    source_width: int
    source_height: int
    exact_duplicate: bool
    near_duplicate_warning: bool
    status: str


@dataclass
class ExtractionStats:
    total_source_images: int = 0
    readable_source_images: int = 0
    unreadable_source_images: int = 0
    exact_duplicate_source_images: int = 0
    low_resolution_source_images: int = 0
    total_roi_candidates_generated: int = 0
    total_rejected_crops: int = 0
    total_duplicate_crops: int = 0
    total_unique_candidate_crops: int = 0


CANDIDATE_SPECS = (
    CandidateSpec("upper_left", 0.31, 0.38, 0.36, 0.34),
    CandidateSpec("upper_center", 0.50, 0.36, 0.38, 0.34),
    CandidateSpec("upper_right", 0.69, 0.38, 0.36, 0.34),
    CandidateSpec("lower_left", 0.31, 0.62, 0.36, 0.34),
    CandidateSpec("lower_center", 0.50, 0.64, 0.38, 0.34),
    CandidateSpec("lower_right", 0.69, 0.62, 0.36, 0.34),
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_source_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced" / "healthy_topup" / "synthetic"


def default_roi_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced" / "healthy_topup" / "synthetic_rois"


def default_rejected_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced" / "healthy_topup" / "synthetic_rejected"


def default_reports_dir(root: Path) -> Path:
    return root / "datasets" / "gingivitis" / "classification_balanced" / "reports"


def find_source_images(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return sorted(
        [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_bytes_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("utf-8"))
    digest.update(str(image.size).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def scan_sources(source_dir: Path, min_source_side: int) -> list[SourceInfo]:
    images = find_source_images(source_dir)
    seen_hashes: dict[str, str] = {}
    infos: list[SourceInfo] = []

    for image_path in images:
        info = SourceInfo(path=image_path, readable=False)
        try:
            info.sha256 = file_sha256(image_path)
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)
                info.width, info.height = image.size
            info.readable = True
            info.low_resolution = min(info.width, info.height) < min_source_side
        except Exception as exc:  # pragma: no cover - defensive for damaged files.
            info.error = str(exc)

        if info.sha256:
            original = seen_hashes.get(info.sha256)
            if original:
                info.exact_duplicate = True
                info.duplicate_of = original
            else:
                seen_hashes[info.sha256] = image_path.name
        infos.append(info)

    return infos


def candidate_box(spec: CandidateSpec, source_width: int, source_height: int) -> tuple[int, int, int, int]:
    crop_width = int(round(source_width * spec.width))
    crop_height = int(round(source_height * spec.height))
    crop_width = min(source_width, max(crop_width, 1))
    crop_height = min(source_height, max(crop_height, 1))

    left = int(round(source_width * spec.center_x - crop_width / 2))
    top = int(round(source_height * spec.center_y - crop_height / 2))
    left = max(0, min(source_width - crop_width, left))
    top = max(0, min(source_height - crop_height, top))
    return left, top, left + crop_width, top + crop_height


def intersection_over_union(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_width = max(0, ix2 - ix1)
    inter_height = max(0, iy2 - iy1)
    intersection = inter_width * inter_height
    if intersection == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / float(area_a + area_b - intersection)


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
    sample.thumbnail((128, 128), Image.Resampling.LANCZOS)
    pixels = list(sample.getdata())
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
        if maximum > 145 and saturation < 72 and mean > 135:
            tooth += 1
        if red > 68 and red > green * 1.06 and red > blue * 1.06 and saturation > 18:
            gingiva += 1

    return tooth / total, gingiva / total, dark / total, bright / total


def dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
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


def assess_crop(
    crop: Image.Image,
    min_crop_side: int,
) -> CropAssessment:
    width, height = crop.size
    if min(width, height) < min_crop_side:
        return CropAssessment(status="rejected_small_crop", score=0.0, reason="crop side below minimum")

    gray = crop.convert("L")
    stat = ImageStat.Stat(gray)
    mean = stat.mean[0]
    stddev = stat.stddev[0]
    crop_entropy = entropy(gray)
    tooth_fraction, gingiva_fraction, dark_fraction, bright_fraction = color_fractions(crop)

    if mean < 28:
        return CropAssessment(status="rejected_too_dark", score=0.0, reason="crop is too dark")
    if mean > 242:
        return CropAssessment(status="rejected_too_bright", score=0.0, reason="crop is too bright")
    if stddev < 13 or crop_entropy < 3.0:
        return CropAssessment(status="rejected_blank_or_uniform", score=0.0, reason="low texture or uniform crop")
    if dark_fraction > 0.58:
        return CropAssessment(status="rejected_obscured_or_empty", score=0.0, reason="mostly dark/empty region")
    if bright_fraction > 0.72:
        return CropAssessment(status="rejected_tooth_or_glare_dominant", score=0.0, reason="mostly bright tooth/glare")
    if tooth_fraction < 0.015:
        return CropAssessment(status="rejected_gum_only_or_no_tooth_reference", score=0.0, reason="too little tooth context")
    if gingiva_fraction < 0.025:
        return CropAssessment(status="rejected_tooth_only_or_no_gingiva", score=0.0, reason="too little gingiva context")

    score = (
        min(tooth_fraction / 0.16, 1.0) * 30
        + min(gingiva_fraction / 0.18, 1.0) * 35
        + min(stddev / 64.0, 1.0) * 20
        + min(crop_entropy / 7.0, 1.0) * 15
        - max(0.0, dark_fraction - 0.30) * 20
    )
    return CropAssessment(status="candidate_review_required", score=score)


def choose_candidates(
    image: Image.Image,
    min_crop_side: int,
    max_rois_per_source: int,
) -> list[tuple[CandidateSpec, tuple[int, int, int, int], Image.Image, CropAssessment]]:
    assessed: list[tuple[CandidateSpec, tuple[int, int, int, int], Image.Image, CropAssessment]] = []
    source_width, source_height = image.size

    for spec in CANDIDATE_SPECS:
        box = candidate_box(spec, source_width, source_height)
        crop = image.crop(box)
        assessment = assess_crop(crop, min_crop_side)
        if assessment.status == "candidate_review_required":
            assessed.append((spec, box, crop, assessment))

    assessed.sort(key=lambda item: item[3].score, reverse=True)
    selected: list[tuple[CandidateSpec, tuple[int, int, int, int], Image.Image, CropAssessment]] = []
    for item in assessed:
        box = item[1]
        if all(intersection_over_union(box, chosen[1]) <= 0.45 for chosen in selected):
            selected.append(item)
        if len(selected) >= max_rois_per_source:
            break

    return selected


def safe_save_jpeg(image: Image.Image, path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    image.save(path, format="JPEG", quality=95, subsampling=0, optimize=True)
    return True


def write_extraction_csv(path: Path, records: list[CropRecord]) -> None:
    fieldnames = [
        "source_image",
        "roi_filename",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "crop_width",
        "crop_height",
        "source_width",
        "source_height",
        "exact_duplicate",
        "near_duplicate_warning",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_duplicates_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["duplicate_type", "item", "duplicate_of", "value", "details"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, stats: ExtractionStats) -> None:
    average = (
        stats.total_roi_candidates_generated / stats.readable_source_images
        if stats.readable_source_images
        else 0.0
    )
    lines = [
        "Synthetic Healthy ROI extraction summary",
        "========================================",
        f"total source images: {stats.total_source_images}",
        f"readable source images: {stats.readable_source_images}",
        f"unreadable source images: {stats.unreadable_source_images}",
        f"exact duplicate source images: {stats.exact_duplicate_source_images}",
        f"unusually low resolution source images: {stats.low_resolution_source_images}",
        f"total ROI candidates generated: {stats.total_roi_candidates_generated}",
        f"total rejected crops: {stats.total_rejected_crops}",
        f"total duplicate crops: {stats.total_duplicate_crops}",
        f"total unique candidate crops: {stats.total_unique_candidate_crops}",
        f"average ROIs per source image: {average:.2f}",
        "",
        "These are candidate Healthy ROI crops for manual review only.",
        "No medical validation is implied.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_record(
    source_name: str,
    filename: str,
    box: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
    assessment: CropAssessment,
) -> CropRecord:
    x1, y1, x2, y2 = box
    return CropRecord(
        source_image=source_name,
        roi_filename=filename,
        crop_x1=x1,
        crop_y1=y1,
        crop_x2=x2,
        crop_y2=y2,
        crop_width=x2 - x1,
        crop_height=y2 - y1,
        source_width=source_width,
        source_height=source_height,
        exact_duplicate=assessment.exact_duplicate,
        near_duplicate_warning=assessment.near_duplicate_warning,
        status=assessment.status,
    )


def extract_rois(args: argparse.Namespace) -> ExtractionStats:
    source_dir = Path(args.source_dir)
    roi_dir = Path(args.roi_dir)
    rejected_dir = Path(args.rejected_dir)
    reports_dir = Path(args.reports_dir)

    source_infos = scan_sources(source_dir, args.min_source_side)
    if len(source_infos) != EXPECTED_SOURCE_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_SOURCE_COUNT} source images, found {len(source_infos)} in {source_dir}"
        )

    roi_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records: list[CropRecord] = []
    duplicate_rows: list[dict[str, str]] = []
    seen_crop_hashes: dict[str, str] = {}
    seen_dhashes: list[tuple[int, str]] = []

    stats = ExtractionStats(
        total_source_images=len(source_infos),
        readable_source_images=sum(1 for info in source_infos if info.readable),
        unreadable_source_images=sum(1 for info in source_infos if not info.readable),
        exact_duplicate_source_images=sum(1 for info in source_infos if info.exact_duplicate),
        low_resolution_source_images=sum(1 for info in source_infos if info.low_resolution),
    )

    for info in source_infos:
        if info.exact_duplicate:
            duplicate_rows.append(
                {
                    "duplicate_type": "source_exact_duplicate",
                    "item": info.path.name,
                    "duplicate_of": info.duplicate_of,
                    "value": info.sha256,
                    "details": "source file hash matches an earlier source image",
                }
            )
        if info.low_resolution:
            duplicate_rows.append(
                {
                    "duplicate_type": "source_low_resolution",
                    "item": info.path.name,
                    "duplicate_of": "",
                    "value": f"{info.width}x{info.height}",
                    "details": f"minimum side below {args.min_source_side}px",
                }
            )
        if not info.readable:
            duplicate_rows.append(
                {
                    "duplicate_type": "source_unreadable",
                    "item": info.path.name,
                    "duplicate_of": "",
                    "value": "",
                    "details": info.error,
                }
            )
            continue

        with Image.open(info.path) as source_image:
            source_image = ImageOps.exif_transpose(source_image).convert("RGB")
            chosen = choose_candidates(source_image, args.min_crop_side, args.max_rois_per_source)

            for index, (_spec, box, crop, assessment) in enumerate(chosen, start=1):
                filename = f"{info.path.stem}_roi_{index:03d}.jpg"
                assessment.sha256 = image_bytes_sha256(crop)
                assessment.dhash = dhash(crop)

                exact_match = seen_crop_hashes.get(assessment.sha256)
                if exact_match:
                    assessment.exact_duplicate = True
                    assessment.status = "rejected_exact_duplicate_crop"
                    duplicate_rows.append(
                        {
                            "duplicate_type": "crop_exact_duplicate",
                            "item": filename,
                            "duplicate_of": exact_match,
                            "value": assessment.sha256,
                            "details": "crop pixels exactly match an earlier crop",
                        }
                    )
                else:
                    seen_crop_hashes[assessment.sha256] = filename

                if not assessment.exact_duplicate and assessment.dhash is not None:
                    nearest_name = ""
                    nearest_distance = 65
                    for previous_hash, previous_name in seen_dhashes:
                        distance = hamming_distance(assessment.dhash, previous_hash)
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest_name = previous_name
                    if nearest_distance <= args.near_duplicate_distance:
                        assessment.near_duplicate_warning = True
                        assessment.near_duplicate_of = nearest_name
                        assessment.near_duplicate_distance = nearest_distance
                        duplicate_rows.append(
                            {
                                "duplicate_type": "crop_near_duplicate_warning",
                                "item": filename,
                                "duplicate_of": nearest_name,
                                "value": str(nearest_distance),
                                "details": "dHash distance is within warning threshold",
                            }
                        )
                    seen_dhashes.append((assessment.dhash, filename))

                output_dir = rejected_dir if assessment.exact_duplicate else roi_dir
                saved = safe_save_jpeg(crop, output_dir / filename, args.overwrite)
                if not saved:
                    assessment.status = "skipped_existing_file"

                if assessment.exact_duplicate:
                    stats.total_duplicate_crops += 1
                    stats.total_rejected_crops += 1
                elif assessment.status == "candidate_review_required":
                    stats.total_roi_candidates_generated += 1
                    stats.total_unique_candidate_crops += 1
                else:
                    stats.total_rejected_crops += 1

                records.append(
                    build_record(
                        info.path.name,
                        filename,
                        box,
                        source_image.width,
                        source_image.height,
                        assessment,
                    )
                )

    write_extraction_csv(reports_dir / "synthetic_roi_extraction.csv", records)
    write_duplicates_csv(reports_dir / "synthetic_roi_duplicates.csv", duplicate_rows)
    write_summary(reports_dir / "synthetic_roi_summary.txt", stats)
    return stats


def print_plan(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    roi_dir = Path(args.roi_dir)
    rejected_dir = Path(args.rejected_dir)
    reports_dir = Path(args.reports_dir)
    source_infos = scan_sources(source_dir, args.min_source_side)
    readable = sum(1 for info in source_infos if info.readable)
    unreadable = sum(1 for info in source_infos if not info.readable)
    exact_duplicates = sum(1 for info in source_infos if info.exact_duplicate)
    low_resolution = sum(1 for info in source_infos if info.low_resolution)
    expected_min = min(len(source_infos) * 2, len(source_infos) * args.max_rois_per_source)
    expected_max = len(source_infos) * args.max_rois_per_source

    print("Synthetic Healthy ROI preflight")
    print("===============================")
    print(f"source folder: {source_dir}")
    print(f"source folder exists: {source_dir.exists()}")
    print(f"source image count: {len(source_infos)}")
    print(f"expected source image count: {EXPECTED_SOURCE_COUNT}")
    print(f"count confirmed: {len(source_infos) == EXPECTED_SOURCE_COUNT}")
    print(f"readable source images: {readable}")
    print(f"unreadable source images: {unreadable}")
    print(f"exact duplicate source images: {exact_duplicates}")
    print(f"unusually low resolution source images: {low_resolution}")
    print()
    print("planned output folders:")
    print(f"  candidate ROIs: {roi_dir}")
    print(f"  rejected/duplicates: {rejected_dir}")
    print(f"  reports: {reports_dir}")
    print()
    print("crop strategy:")
    print("  evaluate six geometry-based regions per image: upper/lower left, center, right")
    print("  keep up to four distinct tooth-plus-gingiva crops per image")
    print("  require minimum crop size, texture, tooth-color context, gingiva-color context")
    print("  reject dark, blank, tooth-only, gum-only, duplicate, and tiny crops")
    print("  warn on near duplicates by perceptual dHash distance")
    print()
    print(f"expected candidate crop range before quality rejection: about {expected_min}-{expected_max}")
    print("manual review remains required; no crop is medically validated or automatically approved")
    print()
    print("No crops were generated. Re-run with --execute after confirmation.")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(default_source_dir(root)))
    parser.add_argument("--roi-dir", default=str(default_roi_dir(root)))
    parser.add_argument("--rejected-dir", default=str(default_rejected_dir(root)))
    parser.add_argument("--reports-dir", default=str(default_reports_dir(root)))
    parser.add_argument("--min-source-side", type=int, default=DEFAULT_MIN_SOURCE_SIDE)
    parser.add_argument("--min-crop-side", type=int, default=DEFAULT_MIN_CROP_SIDE)
    parser.add_argument("--max-rois-per-source", type=int, default=DEFAULT_MAX_ROIS_PER_SOURCE)
    parser.add_argument("--near-duplicate-distance", type=int, default=DEFAULT_NEAR_DUPLICATE_DISTANCE)
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing generated ROI files")
    parser.add_argument("--execute", action="store_true", help="write crops and reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        print_plan(args)
        return

    stats = extract_rois(args)
    print("Synthetic Healthy ROI extraction complete")
    print(f"total ROI candidates generated: {stats.total_roi_candidates_generated}")
    print(f"total rejected crops: {stats.total_rejected_crops}")
    print(f"total duplicate crops: {stats.total_duplicate_crops}")
    print(f"total unique candidate crops: {stats.total_unique_candidate_crops}")


if __name__ == "__main__":
    main()
