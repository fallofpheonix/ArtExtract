from __future__ import annotations

import math
from typing import List, Set

import numpy as np


def evaluate_retrieval(
    emb: np.ndarray, 
    labels: np.ndarray, 
    k: int = 10
) -> dict:
    """
    Compute semantic retrieval metrics (P@K, R@K, mAP@K, nDCG@K).
    
    Excludes 'unknown' labels and singleton classes.
    """
    import pandas as pd
    
    counts = pd.Series(labels).value_counts()
    valid = np.array([(x != "unknown") and (counts.get(x, 0) > 1) for x in labels])
    queries = np.where(valid)[0]
    if len(queries) == 0:
        return {}

    p_sum = 0.0
    r_sum = 0.0
    map_sum = 0.0
    ndcg_sum = 0.0

    for qi in queries:
        # Cosine scores because embeddings are assumed L2-normalized
        s = emb @ emb[qi]
        s[qi] = -np.inf
        top = np.argpartition(-s, kth=min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        retrieved = [int(j) for j in top.tolist()]
        
        relevant_indices = np.where(labels == labels[qi])[0]
        relevant = set(relevant_indices.tolist())
        relevant.discard(int(qi))
        
        if not relevant:
            continue

        hits = 0
        ap = 0.0
        gains = []
        for rank, j in enumerate(retrieved, start=1):
            rel = int(j in relevant)
            gains.append(float(rel))
            if rel:
                hits += 1
                ap += hits / rank

        p = hits / k
        r = hits / len(relevant)
        ap = ap / min(len(relevant), k)
        
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        p_sum += p
        r_sum += r
        map_sum += ap
        ndcg_sum += ndcg

    n = float(len(queries))
    return {
        "queries": int(len(queries)),
        "P@K": p_sum / n,
        "R@K": r_sum / n,
        "mAP@K": map_sum / n,
        "nDCG@K": ndcg_sum / n,
    }
