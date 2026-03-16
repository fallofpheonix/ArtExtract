from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .semantic import CLIPRetriever
from .index import FaissIndex, kmeans_clustering
from .metrics import evaluate_retrieval


def detect_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def objectid_from_filename(path: Path) -> Optional[int]:
    m = re.match(r"^(\d+)", path.stem)
    return int(m.group(1)) if m else None


def build_table(images_dir: Path, opendata_dir: Path, label_col: str, max_images: int) -> pd.DataFrame:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in exts])
    
    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise RuntimeError(f"No images found in {images_dir.resolve()}")

    manifest = pd.DataFrame(
        {"path": [str(p) for p in paths], "objectid": [objectid_from_filename(p) for p in paths]}
    )

    objects_csv = opendata_dir / "data" / "objects.csv"
    published_csv = opendata_dir / "data" / "published_images.csv"
    
    if not objects_csv.exists() or not published_csv.exists():
        manifest["title"] = manifest["path"].map(lambda p: Path(p).name)
        manifest["artist"] = "unknown"
        manifest["classification"] = "unknown"
        manifest["style"] = "unknown"
        return manifest

    obj = pd.read_csv(objects_csv, low_memory=False)
    pub = pd.read_csv(published_csv, low_memory=False)

    pid = detect_col(pub.columns.tolist(), ["objectid", "depictstmsobjectid"])
    if pid and pid != "objectid":
        pub = pub.rename(columns={pid: "objectid"})

    title_col = detect_col(obj.columns.tolist(), ["title"]) or "title"
    artist_col = detect_col(obj.columns.tolist(), ["attribution", "displayname", "artist"]) or "attribution"
    class_col = detect_col(obj.columns.tolist(), ["classification"]) or "classification"
    style_col = detect_col(obj.columns.tolist(), ["style"]) or "style"

    cols = ["objectid"]
    for c in [title_col, artist_col, class_col, style_col]:
        if c in obj.columns and c not in cols:
            cols.append(c)

    out = manifest.merge(obj[cols], on="objectid", how="left")
    out = out.rename(
        columns={
            title_col: "title",
            artist_col: "artist",
            class_col: "classification",
            style_col: "style",
        }
    )
    for c in ["title", "artist", "classification", "style"]:
        if c not in out.columns:
            out[c] = "unknown"
        out[c] = out[c].fillna("unknown").astype(str)

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="CLIP + FAISS painting retrieval pipeline")
    p.add_argument("--images-dir", default="images")
    p.add_argument("--opendata-dir", default="nga_data")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-images", type=int, default=1200)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--label-col", default="classification")
    p.add_argument("--out-dir", default="analysis_out")
    p.add_argument("--clusters", type=int, default=20)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = CLIPRetriever(device=device)
    
    print(f"Embedding images from {args.images_dir}...")
    table = build_table(Path(args.images_dir), Path(args.opendata_dir), args.label_col, args.max_images)
    emb, kept = retriever.embed_images(table["path"].tolist(), batch_size=args.batch_size)
    table = table.iloc[kept].reset_index(drop=True)

    index = FaissIndex(emb.shape[1])
    index.add(emb)
    print(f"Indexed {len(table)} images.")

    metrics = evaluate_retrieval(emb, table[args.label_col].values, k=args.top_k)
    if metrics:
        print(f"Metrics: {metrics}")

    if args.clusters > 1:
        cluster_ids, _ = kmeans_clustering(emb, args.clusters)
        table["cluster"] = cluster_ids
        table.to_csv(out_dir / "embedding_metadata.csv", index=False)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
