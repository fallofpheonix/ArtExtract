from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int = 1) -> float:
    if logits.ndim != 2:
        raise ValueError("logits must be shape [N, C]")
    if y_true.ndim != 1:
        raise ValueError("y_true must be shape [N]")
    if logits.shape[0] != y_true.shape[0]:
        raise ValueError("N mismatch")

    k = min(k, logits.shape[1])
    topk = np.argpartition(logits, -k, axis=1)[:, -k:]
    hit = (topk == y_true[:, None]).any(axis=1)
    return float(hit.mean())


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    yt = np.asarray(list(y_true), dtype=np.int64)
    yp = np.asarray(list(y_pred), dtype=np.int64)
    if yt.shape != yp.shape:
        raise ValueError("y_true/y_pred shape mismatch")

    labels = np.unique(np.concatenate([yt, yp]))
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_support = []

    for c in labels:
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        support = int((yt == c).sum())

        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0

        per_class_precision.append(p)
        per_class_recall.append(r)
        per_class_f1.append(f1)
        per_class_support.append(support)

    macro_precision = float(np.mean(per_class_precision)) if per_class_precision else 0.0
    macro_recall = float(np.mean(per_class_recall)) if per_class_recall else 0.0
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    support = np.asarray(per_class_support, dtype=np.float64)
    weights = support / support.sum() if support.sum() > 0 else np.zeros_like(support)
    weighted_f1 = float((weights * np.asarray(per_class_f1, dtype=np.float64)).sum())

    acc = float((yt == yp).mean()) if yt.size > 0 else 0.0
    balanced_acc = macro_recall

    return {
        "accuracy_top1": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
    }
