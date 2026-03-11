#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artextract.training.outliers import classwise_isolation_outliers
from artextract.utils import ensure_parent_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Classwise embedding outlier detection")
    p.add_argument("--embeddings", required=True, help=".npy [N,D]")
    p.add_argument("--labels", required=True, help=".npy [N]")
    p.add_argument("--out", required=True)
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument("--min-samples-per-class", type=int, default=20)
    args = p.parse_args()

    emb = np.load(args.embeddings)
    y = np.load(args.labels)
    out = classwise_isolation_outliers(
        emb,
        y,
        contamination=args.contamination,
        min_samples_per_class=args.min_samples_per_class,
    )

    ensure_parent_dir(args.out)
    with Path(args.out).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "class_id", "score"])
        for o in out:
            w.writerow([o.index, o.class_id, o.score])

    print(f"wrote {len(out)} outliers -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
