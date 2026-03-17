from __future__ import annotations

import argparse
from pathlib import Path

from artextract.config import RetrievalRuntimeConfig
from artextract.services import SimilarityRetrievalService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIP + FAISS painting retrieval pipeline")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--opendata-dir", default="nga_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=1200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label-col", default="classification")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--clusters", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runtime = RetrievalRuntimeConfig.with_env_defaults(
        images_dir=Path(args.images_dir),
        opendata_dir=Path(args.opendata_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        batch_size=args.batch_size,
        max_images=args.max_images,
        top_k=args.top_k,
        label_col=args.label_col,
        clusters=args.clusters,
    )

    result = SimilarityRetrievalService(runtime).run()
    print(f"Indexed {result['indexed']} images")
    if result["metrics"]:
        print(f"Metrics: {result['metrics']}")
    print(f"Metadata: {result['metadata_path']}")
    return 0
