from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1.0) -> float:
    y_true_f = y_true.reshape(-1).astype(np.float32)
    y_pred_f = y_pred.reshape(-1).astype(np.float32)
    intersection = np.sum(y_true_f * y_pred_f)
    return float((2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def bone_loss_mae(y_true_mm: np.ndarray, y_pred_mm: np.ndarray) -> float:
    return float(mean_absolute_error(y_true_mm, y_pred_mm))


def severity_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred))
