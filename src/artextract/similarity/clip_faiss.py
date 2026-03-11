#!/usr/bin/env python3
"""
End-to-end ArtExtract Task-2 pipeline:
metadata -> CLIP embeddings -> FAISS search -> retrieval metrics.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

# Runtime stability for mixed torch/faiss/openmp stacks on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLIP + FAISS painting retrieval pipeline")
    p.add_argument("--images-dir", default="images", help="Directory with downloaded images")
    p.add_argument("--opendata-dir", default="nga_data", help="Path to NGA opendata clone")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-images", type=int, default=1200, help="Max local images to embed (0 = all)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--label-col",
        default="classification",
        choices=["classification", "style", "artist"],
        help="Label used for relevance in metrics",
    )
    p.add_argument("--query-index", type=int, default=0, help="Query index for demo retrieval")
    p.add_argument("--clusters", type=int, default=20, help="KMeans clusters for style discovery (0 disables)")
    p.add_argument("--tsne-sample", type=int, default=1500, help="Max samples for t-SNE plot")
    p.add_argument("--out-dir", default="analysis_out", help="Directory for analysis artifacts")
    p.add_argument(
        "--clip-backend",
        default="open_clip",
        choices=["auto", "openai", "open_clip"],
        help="CLIP runtime backend selection",
    )
    p.add_argument(
        "--clip-model",
        default="ViT-B/32",
        help="Model name for OpenAI CLIP backend (default: ViT-B/32)",
    )
    p.add_argument(
        "--open-clip-pretrained",
        default="laion2b_s34b_b79k",
        help="Pretrained tag for open_clip backend (default: laion2b_s34b_b79k)",
    )
    return p.parse_args()


def detect_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def objectid_from_filename(path: Path) -> Optional[int]:
    m = re.match(r"^(\d+)", path.stem)
    return int(m.group(1)) if m else None


def choose_device() -> str:
    forced = os.environ.get("FORCE_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda", "mps"}:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    # Default to CPU on macOS unless explicitly forced to MPS.
    if os.environ.get("ENABLE_MPS", "").strip() == "1" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip(args: argparse.Namespace, device: str):
    backend_req = args.clip_backend
    model_name_openai = args.clip_model
    model_name_open_clip = model_name_openai.replace("/", "-")
    errors: List[str] = []

    def _try_openai():
        import clip  # type: ignore

        model, preprocess = clip.load(model_name_openai, device=device)
        model.eval()
        return model, preprocess, f"openai-clip:{model_name_openai}"

    def _try_open_clip():
        # Avoid xet transport stalls in constrained networks.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        import open_clip  # type: ignore

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name_open_clip,
            pretrained=args.open_clip_pretrained,
            device=device,
            cache_dir=str(Path(".hf_cache/hub")),
        )
        model.eval()
        return model, preprocess, f"open-clip:{model_name_open_clip}:{args.open_clip_pretrained}"

    if backend_req in {"openai"}:
        try:
            return _try_openai()
        except Exception as e:  # pragma: no cover
            errors.append(f"openai-clip failed: {type(e).__name__}: {e}")
            if backend_req == "openai":
                raise RuntimeError(errors[-1]) from e

    if backend_req in {"open_clip", "auto"}:
        try:
            return _try_open_clip()
        except Exception as e:  # pragma: no cover
            errors.append(f"open_clip failed: {type(e).__name__}: {e}")
            if backend_req == "open_clip":
                raise RuntimeError(errors[-1]) from e

    if backend_req == "auto":
        try:
            return _try_openai()
        except Exception as e:  # pragma: no cover
            errors.append(f"openai-clip failed: {type(e).__name__}: {e}")

    raise RuntimeError("Unable to load CLIP backend.\n" + "\n".join(errors))


def build_table(images_dir: Path, opendata_dir: Path, label_col: str, max_images: int) -> pd.DataFrame:
    paths = sorted(
        [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    )
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
        # Retrieval can still run without metadata; metrics will be skipped.
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

    # Keep only rows with usable labels for chosen evaluation column when possible.
    if label_col in out.columns:
        out[label_col] = out[label_col].fillna("unknown").astype(str)
    return out


@torch.inference_mode()
def embed_paths(
    paths: List[str], model, preprocess, device: str, batch_size: int
) -> Tuple[np.ndarray, List[int]]:
    feats = []
    kept = []
    for start in tqdm(range(0, len(paths), batch_size), desc="Embedding"):
        cur = paths[start : start + batch_size]
        batch = []
        loc = []
        for j, p in enumerate(cur):
            try:
                batch.append(preprocess(Image.open(p).convert("RGB")))
                loc.append(start + j)
            except (FileNotFoundError, OSError, UnidentifiedImageError):
                continue
        if not batch:
            continue
        x = torch.stack(batch).to(device)
        y = model.encode_image(x)
        y = y / y.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feats.append(y.detach().cpu().numpy().astype("float32"))
        kept.extend(loc)
    if not feats:
        raise RuntimeError("No embeddings generated")
    emb = np.vstack(feats)
    faiss.normalize_L2(emb)
    return emb, kept


def evaluate(emb: np.ndarray, labels: np.ndarray, k: int) -> dict:
    # Exclude unknown labels and singleton classes for stable retrieval metrics.
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
        # Cosine scores because embeddings are L2-normalized.
        s = emb @ emb[qi]
        s[qi] = -np.inf
        top = np.argpartition(-s, kth=min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        retrieved = [int(j) for j in top.tolist()]
        relevant = set(np.where(labels == labels[qi])[0].tolist())
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


def kmeans_cosine(
    x: np.ndarray, k: int, iters: int = 30, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    if k <= 1 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}")
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    faiss.normalize_L2(centers)
    assign = np.zeros(n, dtype=np.int32)

    for _ in range(iters):
        sims = x @ centers.T
        new_assign = np.argmax(sims, axis=1).astype(np.int32)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(k):
            mask = assign == c
            if np.any(mask):
                m = x[mask].mean(axis=0)
            else:
                m = x[rng.integers(0, n)]
            norm = np.linalg.norm(m)
            centers[c] = m / max(norm, 1e-12)
    return assign, centers


def pca_2d(x: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=0, keepdims=True)
    # First two right-singular vectors for 2D projection.
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    w = vt[:2].T
    return x0 @ w


def run_clustering(
    table: pd.DataFrame,
    emb: np.ndarray,
    index: faiss.Index,
    k_clusters: int,
    label_col: str,
    tsne_sample: int,
    out_dir: Path,
) -> None:
    if k_clusters <= 1:
        print("Clustering disabled.")
        return
    if len(table) < k_clusters:
        print(f"Clustering skipped: samples={len(table)} < clusters={k_clusters}")
        return

    cluster_ids, centers = kmeans_cosine(emb, k_clusters, iters=30, seed=42)
    table["cluster"] = cluster_ids

    counts = table.groupby("cluster").size().rename("count").reset_index()
    counts.to_csv(out_dir / "cluster_counts.csv", index=False)

    # Cluster representatives: nearest neighbor to each centroid via FAISS.
    centers = centers.astype("float32")
    faiss.normalize_L2(centers)
    D, I = index.search(centers, 1)
    reps = []
    for c in range(k_clusters):
        j = int(I[c][0])
        reps.append(
            {
                "cluster": c,
                "rep_index": j,
                "rep_score": float(D[c][0]),
                "path": table.iloc[j]["path"],
                "objectid": table.iloc[j].get("objectid"),
                "title": table.iloc[j].get("title", ""),
                "label": table.iloc[j].get(label_col, "unknown"),
            }
        )
    pd.DataFrame(reps).to_csv(out_dir / "cluster_representatives.csv", index=False)
    print(f"Clustering complete: K={k_clusters}")

    # Cluster-label composition.
    if label_col in table.columns:
        comp = (
            table.groupby(["cluster", label_col]).size().rename("count").reset_index()
            .sort_values(["cluster", "count"], ascending=[True, False])
        )
        comp.to_csv(out_dir / "cluster_label_composition.csv", index=False)

    # 2D PCA artifact for embedding inspection.
    n = min(tsne_sample, len(table))
    if n >= 10:
        idx = np.random.default_rng(42).choice(len(table), size=n, replace=False)
        x = emb[idx]
        c = cluster_ids[idx]
        z = pca_2d(x)
        plt.figure(figsize=(9, 7))
        plt.scatter(z[:, 0], z[:, 1], c=c, s=8, alpha=0.75, cmap="tab20")
        plt.title(f"PCA of CLIP embeddings (n={n}, clusters={k_clusters})")
        plt.tight_layout()
        plt.savefig(out_dir / "pca_clusters.png", dpi=140)
        plt.close()
        print(f"PCA plot saved: {out_dir / 'pca_clusters.png'}")
    else:
        print("PCA plot skipped: insufficient samples.")


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir)
    opendata_dir = Path(args.opendata_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    model, preprocess, backend = load_clip(args, device)
    print(f"Device={device} | CLIP backend={backend}")

    table = build_table(images_dir, opendata_dir, args.label_col, args.max_images)
    emb, kept = embed_paths(table["path"].tolist(), model, preprocess, device, args.batch_size)
    table = table.iloc[kept].reset_index(drop=True)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    print(f"Images indexed: {index.ntotal:,} | Embedding dim: {emb.shape[1]}")

    q = max(0, min(args.query_index, len(table) - 1))
    D, I = index.search(emb[q : q + 1], args.top_k + 1)
    print("\nTop matches:")
    for rank, (j, s) in enumerate(zip(I[0], D[0]), start=1):
        if int(j) == q:
            continue
        row = table.iloc[int(j)]
        print(f"{rank:2d}. score={s:.4f} | objectid={row.get('objectid')} | title={row.get('title','')[:80]}")

    labels = table[args.label_col].astype(str).to_numpy()
    metrics = evaluate(emb, labels, args.top_k)
    if metrics:
        print(
            "\nMetrics "
            f"(label={args.label_col}, K={args.top_k}, queries={metrics['queries']}): "
            f"P@K={metrics['P@K']:.4f} "
            f"R@K={metrics['R@K']:.4f} "
            f"mAP@K={metrics['mAP@K']:.4f} "
            f"nDCG@K={metrics['nDCG@K']:.4f}"
        )
    else:
        print("\nMetrics skipped: no valid repeated labels available.")

    run_clustering(
        table=table,
        emb=emb,
        index=index,
        k_clusters=args.clusters,
        label_col=args.label_col,
        tsne_sample=args.tsne_sample,
        out_dir=out_dir,
    )
    table.to_csv(out_dir / "embedding_metadata.csv", index=False)
    print(f"Artifacts written to: {out_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
