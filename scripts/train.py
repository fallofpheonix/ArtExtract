#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artextract.config import load_config
from artextract.data import generate_synthetic_multispectral_dataset
from artextract.training import train_multispectral


def _parse_csv_arg(v: str) -> list[str]:
    return [x.strip().lower() for x in v.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="Unified multispectral trainer")
    p.add_argument("--manifest", default="", help="CSV/JSONL multispectral manifest")
    p.add_argument("--channels", default="rgb,ir,uv,xray")
    p.add_argument("--tasks", default="properties,hidden")
    p.add_argument("--config", default="configs/multispectral_baseline.json")
    p.add_argument("--out-dir", default="")
    p.add_argument("--device", default=None)
    p.add_argument("--synthetic-images-root", default="")
    p.add_argument("--synthetic-out-root", default="data/synthetic_multispectral")
    p.add_argument("--synthetic-max-samples", type=int, default=200)
    args = p.parse_args()

    cfg = load_config(args.config).payload
    channels = _parse_csv_arg(args.channels)
    tasks = _parse_csv_arg(args.tasks)

    manifest = args.manifest.strip()
    if not manifest:
        images_root = args.synthetic_images_root.strip()
        if not images_root:
            print("error: --manifest is required unless --synthetic-images-root is provided", file=sys.stderr)
            return 2
        manifests = generate_synthetic_multispectral_dataset(
            images_root=images_root,
            out_root=args.synthetic_out_root,
            channels=channels,
            image_size=int(cfg.get("model", {}).get("image_size", 128)),
            max_samples=int(args.synthetic_max_samples),
            train_ratio=float(cfg.get("training", {}).get("train_ratio", 0.8)),
            seed=int(cfg.get("seed", 42)),
        )
        manifest = str(manifests.csv_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir.strip() or str(Path("reports") / run_id)

    artifacts = train_multispectral(
        manifest_path=manifest,
        channels=channels,
        tasks=tasks,
        cfg=cfg,
        out_dir=out_dir,
        device=args.device,
    )

    print(json.dumps(
        {
            "out_dir": str(artifacts.out_dir),
            "best_checkpoint": str(artifacts.best_checkpoint),
            "last_checkpoint": str(artifacts.last_checkpoint),
            "history": str(artifacts.history_path),
            "run_meta": str(artifacts.run_meta_path),
            "resolved_manifest": str(artifacts.resolved_manifest_path),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
