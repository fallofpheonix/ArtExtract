from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RetrievalRuntimeConfig:
    images_dir: Path
    opendata_dir: Path
    out_dir: Path
    batch_size: int = 64
    max_images: int = 1200
    top_k: int = 10
    label_col: str = "classification"
    clusters: int = 20

    @classmethod
    def with_env_defaults(
        cls,
        *,
        images_dir: Path,
        opendata_dir: Path,
        out_dir: Path | None = None,
        batch_size: int = 64,
        max_images: int = 1200,
        top_k: int = 10,
        label_col: str = "classification",
        clusters: int = 20,
    ) -> "RetrievalRuntimeConfig":
        base_out = out_dir or Path(os.getenv("ARTEXTRACT_OUTPUT_DIR", "analysis_out"))
        return cls(
            images_dir=images_dir,
            opendata_dir=opendata_dir,
            out_dir=base_out,
            batch_size=batch_size,
            max_images=max_images,
            top_k=top_k,
            label_col=label_col,
            clusters=clusters,
        )
