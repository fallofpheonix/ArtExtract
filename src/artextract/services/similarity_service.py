from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Callable

import torch

from artextract.config import RetrievalRuntimeConfig
from artextract.core import build_manifest_table


class SimilarityRetrievalService:
    def __init__(
        self,
        runtime: RetrievalRuntimeConfig,
        retriever_factory: Callable[..., Any] | None = None,
    ):
        self.runtime = runtime
        self._retriever_factory = retriever_factory

    def run(self) -> dict[str, Any]:
        try:
            from artextract.retrieval.index import FaissIndex, kmeans_clustering
            index_backend = "faiss"
        except ModuleNotFoundError:
            FaissIndex = None
            kmeans_clustering = None
            index_backend = "none"

        from artextract.retrieval.metrics import evaluate_retrieval

        config = self.runtime
        if config.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        config.out_dir.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._retriever_factory is None:
            try:
                from artextract.retrieval.semantic import CLIPRetriever
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "retrieval runtime dependency missing; install requirements.txt"
                ) from exc
            retriever = CLIPRetriever(device=device)
        else:
            retriever = self._retriever_factory(device=device)

        manifest = build_manifest_table(
            images_dir=config.images_dir,
            opendata_dir=config.opendata_dir,
            max_images=config.max_images,
        )
        embeddings, kept_indexes = retriever.embed_images(
            manifest["path"].tolist(),
            batch_size=config.batch_size,
        )
        manifest = manifest.iloc[kept_indexes].reset_index(drop=True)

        if manifest.empty:
            raise RuntimeError("Embedding finished but no rows were kept")

        if FaissIndex is not None:
            index = FaissIndex(embeddings.shape[1])
            index.add(embeddings)

        if config.label_col not in manifest.columns:
            # Why: job should still finish for quick local experiments.
            manifest[config.label_col] = "unknown"

        metrics = evaluate_retrieval(embeddings, manifest[config.label_col].values, k=config.top_k)

        cluster_count = 0
        if kmeans_clustering is not None and config.clusters > 1 and len(manifest) >= config.clusters:
            cluster_ids, _ = kmeans_clustering(embeddings, config.clusters)
            manifest["cluster"] = cluster_ids
            cluster_count = config.clusters
        elif config.clusters > 1 and len(manifest) < config.clusters:
            manifest["cluster"] = -1

        output_file = config.out_dir / "embedding_metadata.csv"
        manifest.to_csv(output_file, index=False)

        return {
            "indexed": int(len(manifest)),
            "metrics": metrics,
            "metadata_path": str(output_file),
            "clusters": cluster_count,
            "index_backend": index_backend,
        }
