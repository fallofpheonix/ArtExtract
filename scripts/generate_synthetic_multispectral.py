#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artextract.data import generate_synthetic_multispectral_dataset


def main() -> int:
    p = argparse.ArgumentParser(description="Generate synthetic multispectral harness dataset")
    p.add_argument("--images-root", required=True)
    p.add_argument("--out-root", default="data/synthetic_multispectral")
    p.add_argument("--channels", default="rgb,ir,uv,xray")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    channels = [c.strip().lower() for c in args.channels.split(",") if c.strip()]
    out = generate_synthetic_multispectral_dataset(
        images_root=args.images_root,
        out_root=args.out_root,
        channels=channels,
        image_size=args.image_size,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print(json.dumps({"manifest_csv": str(out.csv_path), "manifest_jsonl": str(out.jsonl_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
