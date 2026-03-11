from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class OutlierResult:
    index: int
    class_id: int
    score: float


def confidence_outliers(prob: np.ndarray, y_true: np.ndarray, top_n: int = 100) -> List[int]:
    if prob.ndim != 2:
        raise ValueError("prob must be [N, C]")
    idx = np.arange(prob.shape[0])
    conf = prob[idx, y_true]
    order = np.argsort(conf)
    return order[: min(top_n, len(order))].tolist()


def classwise_isolation_outliers(
    emb: np.ndarray,
    y: np.ndarray,
    contamination: float = 0.02,
    min_samples_per_class: int = 20,
) -> List[OutlierResult]:
    """Numpy-only classwise outlier scoring via centroid distance.

    score = z-scored distance from class centroid (higher distance => stronger outlier).
    Returns top `contamination` fraction per class.

    Time complexity: O(N*D)
    Space complexity: O(N + D)
    """
    if emb.ndim != 2:
        raise ValueError("emb must be [N, D]")
    if y.ndim != 1 or y.shape[0] != emb.shape[0]:
        raise ValueError("y must be [N] matching emb")

    out: List[OutlierResult] = []
    frac = max(0.0, min(0.5, float(contamination)))

    for c in np.unique(y):
        cls_idx = np.where(y == c)[0]
        if cls_idx.size < min_samples_per_class:
            continue

        x = emb[cls_idx]
        mu = x.mean(axis=0, keepdims=True)
        d = np.linalg.norm(x - mu, axis=1)

        m = float(d.mean())
        s = float(d.std())
        z = (d - m) / (s + 1e-12)

        k = max(1, int(np.ceil(frac * cls_idx.size)))
        pick = np.argsort(z)[-k:]

        for local in pick:
            out.append(
                OutlierResult(index=int(cls_idx[local]), class_id=int(c), score=float(z[local]))
            )

    out.sort(key=lambda x: x.score, reverse=True)
    return out
