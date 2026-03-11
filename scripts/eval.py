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

from artextract.config import load_config
from artextract.evaluation import evaluate_multispectral


def _parse_csv_arg(v: str) -> list[str]:
    return [x.strip().lower() for x in v.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="Unified multispectral evaluator")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pigments-vocab", required=True)
    p.add_argument("--channels", default="rgb,ir,uv,xray")
    p.add_argument("--tasks", default="properties,hidden")
    p.add_argument("--config", default="configs/multispectral_baseline.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = load_config(args.config).payload
    channels = _parse_csv_arg(args.channels)
    tasks = _parse_csv_arg(args.tasks)

    metrics_path = evaluate_multispectral(
        manifest_path=args.manifest,
        channels=channels,
        tasks=tasks,
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        pigments_vocab_path=args.pigments_vocab,
        device=args.device,
    )

    print(json.dumps({"metrics": str(metrics_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
