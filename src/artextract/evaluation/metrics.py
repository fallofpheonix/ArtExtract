from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def binary_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    cm = binary_confusion(y_true, y_pred)
    tp = float(cm[1, 1])
    fp = float(cm[0, 1])
    fn = float(cm[1, 0])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def multi_label_f1(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1.0 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1.0 - y_pred)).sum(axis=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return float(f1.mean())


def roc_curve_binary(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sort by descending score.
    idx = np.argsort(-y_score)
    y_true = y_true[idx].astype(np.int64)
    y_score = y_score[idx]

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    pos = max(1, int((y_true == 1).sum()))
    neg = max(1, int((y_true == 0).sum()))

    tpr = tps / pos
    fpr = fps / neg

    thresholds = y_score
    # prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[math.inf], thresholds])
    return fpr, tpr, thresholds


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    # x assumed sorted.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0.0:
        return 99.0
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))


def ssim_simple(x: np.ndarray, y: np.ndarray, c1: float = 0.01**2, c2: float = 0.03**2) -> float:
    # Global SSIM approximation (not windowed), deterministic and dependency-light.
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    if den == 0:
        return 0.0
    return float(num / den)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))
