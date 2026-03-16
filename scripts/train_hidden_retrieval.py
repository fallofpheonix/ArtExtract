#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# SRC = ROOT / "src"  # Removed path hack

from artextract.config import load_config
from artextract.reconstruction import (
    SyntheticHiddenRetrievalDataset,
    ReconstructionUNet as UNetRetrieval,
    mae,
    mse,
    psnr,
)
from artextract.utils import ensure_parent_dir

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(arg: str | None) -> "torch.device":
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_preview(observed: "torch.Tensor", target: "torch.Tensor", pred: "torch.Tensor", out_png: Path) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    n = min(3, observed.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(n):
        axes[i, 0].imshow(observed[i].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0))
        axes[i, 0].set_title("Observed")
        axes[i, 1].imshow(pred[i].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0))
        axes[i, 1].set_title("Predicted Hidden")
        axes[i, 2].imshow(target[i].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0))
        axes[i, 2].set_title("Target Hidden")
        for j in range(3):
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _run_epoch(
    model: "UNetRetrieval",
    loader: "DataLoader",
    device: "torch.device",
    optimizer: "torch.optim.Optimizer | None",
    loss_l1_w: float,
    loss_mse_w: float,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    n = 0
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0

    for observed, target, _meta in loader:
        observed = observed.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        pred = model(observed)
        loss_l1 = nn.functional.l1_loss(pred, target)
        loss_mse = nn.functional.mse_loss(pred, target)
        loss = loss_l1_w * loss_l1 + loss_mse_w * loss_mse

        if train_mode:
            loss.backward()
            optimizer.step()

        b = observed.size(0)
        n += b
        total_loss += float(loss.item()) * b
        batch_mae = float(mae(pred, target).item())
        batch_mse = float(mse(pred, target).item())
        total_mae += batch_mae * b
        total_mse += batch_mse * b
        total_psnr += float(psnr(pred, target)) * b

    if n == 0:
        return {"loss": 0.0, "mae": 0.0, "mse": 0.0, "psnr": 0.0}

    return {
        "loss": total_loss / n,
        "mae": total_mae / n,
        "mse": total_mse / n,
        "psnr": total_psnr / n,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Train synthetic hidden-image retrieval model")
    p.add_argument("--config", default="configs/retrieval_baseline.json")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    if _IMPORT_ERROR is not None:
        print(f"error: torch unavailable: {_IMPORT_ERROR}", file=sys.stderr)
        return 2

    cfg = load_config(args.config).payload
    dcfg = cfg.get("data", {})
    mcfg = cfg.get("model", {})
    tcfg = cfg.get("training", {})
    lcfg = cfg.get("loss", {})

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    images_root = dcfg.get("images_root")
    if not images_root:
        print("error: config.data.images_root is required", file=sys.stderr)
        return 2

    epochs = int(args.epochs if args.epochs is not None else tcfg.get("epochs", 10))
    batch_size = int(args.batch_size if args.batch_size is not None else tcfg.get("batch_size", 16))
    lr = float(args.lr if args.lr is not None else tcfg.get("lr", 3e-4))
    num_workers = int(args.num_workers if args.num_workers is not None else tcfg.get("num_workers", 0))
    out_dir = Path(args.out_dir if args.out_dir else tcfg.get("out_dir", "outputs/hidden_retrieval"))
    device = _select_device(args.device or tcfg.get("device"))

    image_size = int(dcfg.get("image_size", 128))
    max_images = int(dcfg.get("max_images", 0))
    train_split = float(dcfg.get("train_split", 0.85))

    train_ds = SyntheticHiddenRetrievalDataset(
        images_root=images_root,
        split="train",
        image_size=image_size,
        train_split=train_split,
        max_images=max_images,
        seed=seed,
    )
    val_ds = SyntheticHiddenRetrievalDataset(
        images_root=images_root,
        split="val",
        image_size=image_size,
        train_split=train_split,
        max_images=max_images,
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
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

    loss_l1_w = float(lcfg.get("l1", 1.0))
    loss_mse_w = float(lcfg.get("mse", 0.5))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    ensure_parent_dir(out_dir / "x")
    history = []
    best_psnr = -1.0

    for epoch in range(1, epochs + 1):
        train_m = _run_epoch(model, train_loader, device, optimizer, loss_l1_w, loss_mse_w)
        with torch.no_grad():
            val_m = _run_epoch(model, val_loader, device, None, loss_l1_w, loss_mse_w)
        scheduler.step()

        row = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"])}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}": v for k, v in val_m.items()})
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f} "
            f"val_mae={val_m['mae']:.4f} val_psnr={val_m['psnr']:.3f}"
        )

        torch.save(model.state_dict(), out_dir / "last_model.pt")
        if val_m["psnr"] > best_psnr:
            best_psnr = val_m["psnr"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    with (out_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # preview from validation
    with torch.no_grad():
        observed, target, _meta = next(iter(val_loader))
        observed = observed.to(device)
        pred = model(observed)
        _save_preview(observed.cpu(), target.cpu(), pred.cpu(), out_dir / "val_preview.png")

    run_meta = {
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "images_root": str(images_root),
        "image_size": image_size,
        "max_images": max_images,
    }
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    final_metrics = {
        "best_val_psnr": best_psnr,
        "last_epoch": history[-1] if history else {},
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"done: best_val_psnr={best_psnr:.3f}")
    print(f"artifacts: {out_dir / 'best_model.pt'} {out_dir / 'metrics.json'} {out_dir / 'val_preview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
