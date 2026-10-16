from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path


SOURCE_ROOT = Path(r"D:\Gingivities Dataset")
TARGET_ROOT = Path(r"D:\Oral_AI_System")


def ignore_runtime_dirs(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in {"__pycache__", "node_modules", "dist"}
        or name.endswith(".pyc")
    }


def copytree_clean(source: Path, target: Path) -> None:
    shutil.copytree(source, target, ignore=ignore_runtime_dirs)


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def create_structure() -> None:
    directories = [
        "backend/app/routes",
        "backend/app/services",
        "backend/app/schemas",
        "frontend",
        "models/gingivitis",
        "models/tooth_discoloration",
        "datasets/gingivitis",
        "datasets/tooth_discoloration",
        "scripts/gingivitis",
        "scripts/tooth_discoloration",
        "outputs/gingivitis",
        "outputs/tooth_discoloration",
        "outputs/combined",
        "reports/gingivitis",
        "reports/tooth_discoloration",
        "archive",
    ]
    for directory in directories:
        (TARGET_ROOT / directory).mkdir(parents=True, exist_ok=True)


def copy_models() -> dict[str, tuple[str, str]]:
    model_pairs = [
        (
            SOURCE_ROOT / "models" / "gingivitis_yolo26s_final_best.pt",
            TARGET_ROOT / "models" / "gingivitis" / "gingivitis_yolo26s_final_best.pt",
        ),
        (
            SOURCE_ROOT / "models" / "efficientnet_b0_stage2_severity_best.pth",
            TARGET_ROOT
            / "models"
            / "gingivitis"
            / "efficientnet_b0_stage2_severity_best.pth",
        ),
    ]
    hashes: dict[str, tuple[str, str]] = {}
    for source, target in model_pairs:
        copy_file(source, target)
        hashes[target.name] = (sha256_file(source), sha256_file(target))
    return hashes


def copy_datasets() -> None:
    copytree_clean(
        SOURCE_ROOT / "datasets" / "yolo26_final",
        TARGET_ROOT / "datasets" / "gingivitis" / "yolo26_final",
    )
    copytree_clean(
        SOURCE_ROOT / "datasets" / "efficientnet_stage2",
        TARGET_ROOT / "datasets" / "gingivitis" / "efficientnet_stage2",
    )


def copy_scripts() -> None:
    target = TARGET_ROOT / "scripts" / "gingivitis"
    for item in (SOURCE_ROOT / "scripts").iterdir():
        if item.name == "__pycache__":
            continue
        if item.is_file():
            copy_file(item, target / item.name)
    source_nested = SOURCE_ROOT / "scripts" / "gingivitis"
    if source_nested.exists():
        for item in source_nested.iterdir():
            if item.name == "__pycache__":
                continue
            destination = target / item.name
            if item.is_dir():
                copytree_clean(item, destination)
            elif item.is_file():
                copy_file(item, destination)


def copy_reports_and_outputs() -> None:
    source_reports = SOURCE_ROOT / "reports"
    if source_reports.exists():
        for item in source_reports.iterdir():
            destination = TARGET_ROOT / "reports" / "gingivitis" / item.name
            if item.is_dir():
                copytree_clean(item, destination)
            elif item.is_file():
                copy_file(item, destination)

    source_outputs = SOURCE_ROOT / "hybrid_outputs"
    if source_outputs.exists():
        for item in source_outputs.iterdir():
            destination = TARGET_ROOT / "outputs" / "gingivitis" / item.name
            if item.is_dir():
                copytree_clean(item, destination)
            elif item.is_file():
                copy_file(item, destination)


def copy_frontend() -> None:
    copytree_clean(SOURCE_ROOT / "frontend", TARGET_ROOT / "frontend")
    pages = TARGET_ROOT / "frontend" / "src" / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    for page in [
        "Overview",
        "Upload",
        "Processing",
        "GingivitisResult",
        "DiscolorationResult",
        "CombinedReport",
    ]:
        write_text(
            pages / f"{page}.jsx",
            f"function {page}() {{\n"
            "  return null;\n"
            "}\n\n"
            f"export default {page};\n",
        )


def create_backend_backup() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = TARGET_ROOT / "archive" / f"runtime_backup_{timestamp}"
    copytree_clean(SOURCE_ROOT / "backend", backup_root / "backend_original")
    copytree_clean(SOURCE_ROOT / "frontend", backup_root / "frontend_original")
    return backup_root


def create_backend() -> None:
    backend_root = TARGET_ROOT / "backend"
    copy_file(SOURCE_ROOT / "backend" / "requirements.txt", backend_root / "requirements.txt")

    service_text = (SOURCE_ROOT / "backend" / "app" / "model_service.py").read_text(
        encoding="utf-8"
    )
    service_text = service_text.replace(
        r'PROJECT_ROOT = Path(r"D:\Gingivities Dataset")',
        r'PROJECT_ROOT = Path(r"D:\Oral_AI_System")',
    )
    service_text = service_text.replace(
        'YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "gingivitis_yolo26s_final_best.pt"',
        'YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "gingivitis" / "gingivitis_yolo26s_final_best.pt"',
    )
    service_text = service_text.replace(
        'PROJECT_ROOT / "models" / "efficientnet_b0_stage2_severity_best.pth"',
        'PROJECT_ROOT\n    / "models"\n    / "gingivitis"\n    / "efficientnet_b0_stage2_severity_best.pth"',
    )

    schemas_text = (SOURCE_ROOT / "backend" / "app" / "schemas.py").read_text(
        encoding="utf-8"
    )

    main_text = '''from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.gingivitis import router as gingivitis_router
from app.services.gingivitis_service import (
    GingivitisHybridModelService,
    ModelLoadError,
)


model_service = GingivitisHybridModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load_models()
    except ModelLoadError:
        pass
    yield


app = FastAPI(
    title="Oral AI System API",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.gingivitis_service = model_service

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gingivitis_router)
'''

    route_text = '''from __future__ import annotations

import traceback

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas.gingivitis import HealthResponse, PredictionResponse
from app.services.gingivitis_service import PredictionError


router = APIRouter(tags=["gingivitis"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict[str, str | None]:
    return request.app.state.gingivitis_service.health()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.45),
    iou_threshold: float = Form(0.45),
    max_detections: int = Form(12),
) -> dict:
    print("[/predict] request received", flush=True)
    print(f"[/predict] uploaded filename: {image.filename}", flush=True)
    print("Received confidence:", confidence_threshold, flush=True)
    print("Received IoU:", iou_threshold, flush=True)
    print("Received max detections:", max_detections, flush=True)

    if not image.filename:
        raise HTTPException(status_code=400, detail="Missing image filename.")

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")
        print("[/predict] prediction starts", flush=True)
        return request.app.state.gingivitis_service.predict(
            image_bytes=image_bytes,
            filename=image.filename,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
    except ValueError as exc:
        print("[/predict] validation error", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PredictionError as exc:
        print("[/predict] prediction error", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        print("[/predict] unexpected exception", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
'''

    for relative in [
        "app/__init__.py",
        "app/routes/__init__.py",
        "app/services/__init__.py",
        "app/schemas/__init__.py",
    ]:
        write_text(backend_root / relative, "")
    write_text(backend_root / "app" / "main.py", main_text)
    write_text(backend_root / "app" / "routes" / "gingivitis.py", route_text)
    write_text(backend_root / "app" / "services" / "gingivitis_service.py", service_text)
    write_text(backend_root / "app" / "schemas" / "gingivitis.py", schemas_text)


def count_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def write_migration_report(model_hashes: dict[str, tuple[str, str]], backup_root: Path) -> None:
    lines = [
        "Oral AI System migration report",
        f"Source root: {SOURCE_ROOT}",
        f"Target root: {TARGET_ROOT}",
        f"Runtime backup: {backup_root}",
        "",
        "Model hash verification",
    ]
    for name, (source_hash, target_hash) in model_hashes.items():
        lines.append(f"  {name}: {'PASS' if source_hash == target_hash else 'FAIL'}")
        lines.append(f"    source: {source_hash}")
        lines.append(f"    target: {target_hash}")

    lines.extend(
        [
            "",
            "Dataset file counts",
            f"  source yolo26_final: {count_files(SOURCE_ROOT / 'datasets' / 'yolo26_final')}",
            f"  target yolo26_final: {count_files(TARGET_ROOT / 'datasets' / 'gingivitis' / 'yolo26_final')}",
            f"  source efficientnet_stage2: {count_files(SOURCE_ROOT / 'datasets' / 'efficientnet_stage2')}",
            f"  target efficientnet_stage2: {count_files(TARGET_ROOT / 'datasets' / 'gingivitis' / 'efficientnet_stage2')}",
            "",
        ]
    )
    write_text(TARGET_ROOT / "reports" / "gingivitis" / "migration_report.txt", "\n".join(lines))


def main() -> int:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Missing source root: {SOURCE_ROOT}")
    if TARGET_ROOT.exists():
        raise FileExistsError(f"Target already exists; refusing to overwrite: {TARGET_ROOT}")

    create_structure()
    backup_root = create_backend_backup()
    model_hashes = copy_models()
    copy_datasets()
    copy_scripts()
    copy_reports_and_outputs()
    copy_frontend()
    create_backend()
    write_migration_report(model_hashes, backup_root)

    print(f"Created {TARGET_ROOT}")
    print(f"Runtime backup: {backup_root}")
    for name, (source_hash, target_hash) in model_hashes.items():
        print(f"{name}: {'PASS' if source_hash == target_hash else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
