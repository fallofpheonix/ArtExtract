from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from artextract.data import (
    MultiSpectralDataset,
    MultiSpectralRecord,
    collect_pigment_vocab,
    load_multispectral_manifest,
    multispectral_collate,
)
from artextract.data.multispectral import write_multispectral_manifest_csv
from artextract.models import MultiSpectralMultiTaskModel

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


@dataclass(frozen=True)
class TrainArtifacts:
    out_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    history_path: Path
    run_meta_path: Path
    resolved_manifest_path: Path


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str | None) -> "torch.device":
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_train_val_splits(
    records: Sequence[MultiSpectralRecord],
    train_ratio: float,
    seed: int,
) -> list[MultiSpectralRecord]:
    has_train = any(r.split == "train" for r in records)
    has_val = any(r.split == "val" for r in records)
    if has_train and has_val:
        return list(records)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_train = max(1, int(len(records) * train_ratio))
    n_train = min(n_train, len(records) - 1)
    train_idx = set(idx[:n_train].tolist())

    out: list[MultiSpectralRecord] = []
    for i, r in enumerate(records):
        split = "train" if i in train_idx else "val"
        out.append(
            MultiSpectralRecord(
                sample_id=r.sample_id,
                split=split,
                channels=r.channels,
                channel_paths=r.channel_paths,
                width=r.width,
                height=r.height,
                pigments=r.pigments,
                damage=r.damage,
                restoration=r.restoration,
                hidden_image=r.hidden_image,
                hidden_gt_path=r.hidden_gt_path,
            )
        )
    return out


def _run_epoch(
    model: "MultiSpectralMultiTaskModel",
    loader: "DataLoader",
    device: "torch.device",
    tasks: set[str],
    loss_weights: Dict[str, float],
    optimizer: "torch.optim.Optimizer | None" = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    n = 0
    sums: Dict[str, float] = {
        "loss_total": 0.0,
        "loss_prop": 0.0,
        "loss_hidden": 0.0,
        "loss_recon": 0.0,
        "hidden_acc": 0.0,
    }

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["channel_mask"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        out = model(x, mask)

        loss_total = x.new_tensor(0.0)
        loss_prop = x.new_tensor(0.0)
        loss_hidden = x.new_tensor(0.0)
        loss_recon = x.new_tensor(0.0)

        if "properties" in tasks:
            l_pig = bce(out["pigments_logits"], targets["pigments"])
            l_damage = bce(out["damage_logits"], targets["damage"])
            l_restore = bce(out["restoration_logits"], targets["restoration"])
            loss_prop = (l_pig + l_damage + l_restore) / 3.0
            loss_total = loss_total + float(loss_weights.get("property", 1.0)) * loss_prop

        if "hidden" in tasks:
            loss_hidden = bce(out["hidden_logits"], targets["hidden_image"])
            loss_total = loss_total + float(loss_weights.get("hidden", 1.0)) * loss_hidden

        if "reconstruction" in tasks:
            l_l1 = l1(out["reconstruction"], targets["hidden_gt"])
            l_mse = mse(out["reconstruction"], targets["hidden_gt"])
            loss_recon = l_l1 + l_mse
            loss_total = loss_total + float(loss_weights.get("reconstruction", 1.0)) * loss_recon

        if train_mode:
            loss_total.backward()
            optimizer.step()

        b = x.size(0)
        n += b
        sums["loss_total"] += float(loss_total.item()) * b
        sums["loss_prop"] += float(loss_prop.item()) * b
        sums["loss_hidden"] += float(loss_hidden.item()) * b
        sums["loss_recon"] += float(loss_recon.item()) * b

        if "hidden" in tasks:
            pred_hidden = (torch.sigmoid(out["hidden_logits"]) >= 0.5).float()
            acc = (pred_hidden == targets["hidden_image"]).float().mean()
            sums["hidden_acc"] += float(acc.item()) * b

    if n == 0:
        return {k: 0.0 for k in sums}

    return {k: v / n for k, v in sums.items()}


def train_multispectral(
    manifest_path: str | Path,
    channels: Sequence[str],
    tasks: Iterable[str],
    cfg: Dict[str, object],
    out_dir: str | Path,
    device: str | None = None,
) -> TrainArtifacts:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"torch unavailable: {_IMPORT_ERROR}")

    tasks_set = {t.strip().lower() for t in tasks}
    if not tasks_set:
        raise ValueError("tasks cannot be empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}

    seed = int(cfg.get("seed", 42))
    set_deterministic(seed)

    records = load_multispectral_manifest(manifest_path)
    records = _ensure_train_val_splits(
        records,
        train_ratio=float(train_cfg.get("train_ratio", 0.8)),
        seed=seed,
    )

    resolved_manifest_path = out_dir / "resolved_manifest.csv"
    write_multispectral_manifest_csv(resolved_manifest_path, records)

    train_records = [r for r in records if r.split == "train"]
    pigments_vocab = collect_pigment_vocab(train_records)
    with (out_dir / "pigments_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(pigments_vocab, f, indent=2)

    image_size = int(model_cfg.get("image_size", 128))

    train_ds = MultiSpectralDataset(
        manifest_path=resolved_manifest_path,
        channels_order=channels,
        split="train",
        tasks=tasks_set,
        pigments_vocab=pigments_vocab,
        image_size=image_size,
        strict_dimensions=bool(model_cfg.get("strict_dimensions", False)),
    )
    val_ds = MultiSpectralDataset(
        manifest_path=resolved_manifest_path,
        channels_order=channels,
        split="val",
        tasks=tasks_set,
        pigments_vocab=pigments_vocab,
        image_size=image_size,
        strict_dimensions=bool(model_cfg.get("strict_dimensions", False)),
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("train/val split produced empty dataset")

    dev = _resolve_device(device or train_cfg.get("device"))
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        collate_fn=multispectral_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        collate_fn=multispectral_collate,
    )

    model = MultiSpectralMultiTaskModel(
        in_channels=len(channels),
        num_pigments=len(pigments_vocab),
        enable_properties=("properties" in tasks_set),
        enable_hidden=("hidden" in tasks_set),
        enable_reconstruction=("reconstruction" in tasks_set),
        base_channels=int(model_cfg.get("base_channels", 32)),
    ).to(dev)

    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(train_cfg.get("epochs", 5))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    loss_weights = cfg.get("loss_weights", {}) if isinstance(cfg.get("loss_weights", {}), dict) else {}
    loss_weights = {
        "property": float(loss_weights.get("property", 1.0)),
        "hidden": float(loss_weights.get("hidden", 1.0)),
        "reconstruction": float(loss_weights.get("reconstruction", 1.0)),
    }

    history = []
    best_val = float("inf")
    ckpt_best = out_dir / "best_model.pt"
    ckpt_last = out_dir / "last_model.pt"

    for epoch in range(1, epochs + 1):
        train_m = _run_epoch(model, train_loader, dev, tasks_set, loss_weights, optimizer=optimizer)
        with torch.no_grad():
            val_m = _run_epoch(model, val_loader, dev, tasks_set, loss_weights, optimizer=None)

        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": float(v) for k, v in train_m.items()},
            **{f"val_{k}": float(v) for k, v in val_m.items()},
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} train_loss={train_m['loss_total']:.4f} "
            f"val_loss={val_m['loss_total']:.4f} "
            f"val_hidden_acc={val_m['hidden_acc']:.4f}"
        )

        torch.save(model.state_dict(), ckpt_last)
        if val_m["loss_total"] < best_val:
            best_val = val_m["loss_total"]
            torch.save(model.state_dict(), ckpt_best)

    history_path = out_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    run_meta = {
        "seed": seed,
        "device": str(dev),
        "channels": list(channels),
        "tasks": sorted(tasks_set),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "loss_weights": loss_weights,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "manifest_path": str(Path(manifest_path).resolve()),
        "resolved_manifest": str(resolved_manifest_path.resolve()),
        "config": cfg,
    }
    run_meta_path = out_dir / "run_meta.json"
    with run_meta_path.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    return TrainArtifacts(
        out_dir=out_dir,
        best_checkpoint=ckpt_best,
        last_checkpoint=ckpt_last,
        history_path=history_path,
        run_meta_path=run_meta_path,
        resolved_manifest_path=resolved_manifest_path,
    )
