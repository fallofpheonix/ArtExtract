#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# SRC = ROOT / "src" # Removed path hack

from artextract.config import load_config
from artextract.reconstruction import (
    SyntheticHiddenRetrievalDataset,
    ReconstructionUNet as UNetRetrieval,
    mae,
    mse,
    psnr,
)

try:
    import torch
    from torch.utils.data import DataLoader
except Exception as e:  # pragma: no cover
    torch = None
    DataLoader = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _select_device(arg: str | None) -> "torch.device":
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate hidden-image retrieval checkpoint")
    p.add_argument("--config", default="configs/retrieval_baseline.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="outputs/hidden_retrieval/val_metrics_eval.json")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if _IMPORT_ERROR is not None:
        print(f"error: torch unavailable: {_IMPORT_ERROR}", file=sys.stderr)
        return 2

    cfg = load_config(args.config).payload
    dcfg = cfg.get("data", {})
    mcfg = cfg.get("model", {})
    tcfg = cfg.get("training", {})

    images_root = dcfg.get("images_root")
    if not images_root:
        print("error: config.data.images_root is required", file=sys.stderr)
        return 2

    device = _select_device(args.device or tcfg.get("device"))
    batch_size = int(args.batch_size if args.batch_size is not None else tcfg.get("batch_size", 16))
    num_workers = int(args.num_workers if args.num_workers is not None else tcfg.get("num_workers", 0))

    val_ds = SyntheticHiddenRetrievalDataset(
        images_root=images_root,
        split="val",
        image_size=int(dcfg.get("image_size", 128)),
        train_split=float(dcfg.get("train_split", 0.85)),
        max_images=int(dcfg.get("max_images", 0)),
        seed=int(cfg.get("seed", 42)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = UNetRetrieval(
        in_ch=int(mcfg.get("in_ch", 3)),
        out_ch=int(mcfg.get("out_ch", 3)),
        base_ch=int(mcfg.get("base_ch", 32)),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    n = 0
    sum_mae = 0.0
    sum_mse = 0.0
    sum_psnr = 0.0

    with torch.no_grad():
        for observed, target, _meta in val_loader:
            observed = observed.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(observed)

            b = observed.size(0)
            n += b
            sum_mae += float(mae(pred, target).item()) * b
            sum_mse += float(mse(pred, target).item()) * b
            sum_psnr += float(psnr(pred, target)) * b

    metrics = {
        "samples_val": n,
        "mae": (sum_mae / n) if n else 0.0,
        "mse": (sum_mse / n) if n else 0.0,
        "psnr": (sum_psnr / n) if n else 0.0,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
