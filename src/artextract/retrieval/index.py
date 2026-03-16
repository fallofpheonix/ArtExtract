from __future__ import annotations

from typing import Tuple

import faiss
import numpy as np

from .base import BaseIndex


class FaissIndex(BaseIndex):
    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dim)
        elif metric == "l2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if not embeddings.flags.c_contiguous:
            embeddings = np.ascontiguousarray(embeddings)
        
        # FAISS expects normalized vectors for Inner Product to simulate Cosine
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        if not queries.flags.c_contiguous:
            queries = np.ascontiguousarray(queries)
        
        faiss.normalize_L2(queries)
        return self.index.search(queries, k)


def kmeans_clustering(
    x: np.ndarray, 
    k: int, 
    iters: int = 30, 
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    if k <= 1 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}")
    
    x = np.ascontiguousarray(x, dtype="float32")
    kmeans = faiss.Kmeans(
        d,
        k,
        niter=iters,
        seed=seed,
        verbose=False,
        spherical=True,
    )
    kmeans.train(x)
    _, assign = kmeans.index.search(x, 1)
    return assign.ravel().astype(np.int32), kmeans.centroids
