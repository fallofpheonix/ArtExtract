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
from artextract.evaluation import evaluate_multispectral
from artextract.training import train_multispectral


def main() -> int:
    p = argparse.ArgumentParser(description="Thin wrapper for hidden+reconstruction multispectral run")
    p.add_argument("--images-root", required=True, help="Source image root for synthetic multispectral harness")
    p.add_argument("--channels", default="rgb,ir,uv,xray")
    p.add_argument("--config", default="configs/multispectral_baseline.json")
    p.add_argument("--out-dir", default="")
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = load_config(args.config).payload
    channels = [x.strip().lower() for x in args.channels.split(",") if x.strip()]
    tasks = ["hidden", "reconstruction"]

    synthetic = generate_synthetic_multispectral_dataset(
        images_root=args.images_root,
        out_root="data/synthetic_multispectral_recon",
        channels=channels,
        image_size=int(cfg.get("model", {}).get("image_size", 128)),
        max_samples=int(args.max_samples),
        train_ratio=float(cfg.get("training", {}).get("train_ratio", 0.8)),
        seed=int(cfg.get("seed", 42)),
    )

    run_id = datetime.now().strftime("reconstruction_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("reports") / run_id

    artifacts = train_multispectral(
        manifest_path=synthetic.csv_path,
        channels=channels,
        tasks=tasks,
        cfg=cfg,
        out_dir=out_dir,
        device=args.device,
    )

    metrics_path = evaluate_multispectral(
        manifest_path=artifacts.resolved_manifest_path,
        channels=channels,
        tasks=tasks,
        cfg=cfg,
        checkpoint_path=artifacts.best_checkpoint,
        out_dir=artifacts.out_dir,
        pigments_vocab_path=artifacts.out_dir / "pigments_vocab.json",
        device=args.device,
    )

    print(json.dumps({
        "out_dir": str(artifacts.out_dir),
        "metrics": str(metrics_path),
        "best_checkpoint": str(artifacts.best_checkpoint),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
