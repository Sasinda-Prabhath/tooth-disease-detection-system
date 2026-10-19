"""Paths anchored to the project, independent of the current working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
DATASET_DIR = PROJECT_ROOT / 'dataset'
RUNS_DIR = PROJECT_ROOT / 'runs'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'evaluation'
UI_ASSETS = PROJECT_ROOT / 'dental_fdi' / 'ui' / 'assets'
DEFAULT_DETECTOR = RUNS_DIR / 'detect' / 'train-2' / 'weights' / 'best.pt'
DEFAULT_ENSEMBLE = RUNS_DIR / 'ensemble' / 'yolo26_effnet_resnet50' / 'model.fdi.zip'
