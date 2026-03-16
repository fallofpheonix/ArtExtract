#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from artextract.config import load_config
from artextract.data import MultiTaskImageDataset, load_class_map
from artextract.utils import ensure_parent_dir

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import transforms
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    WeightedRandomSampler = None
    transforms = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks: int = 3) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))

    def forward(self, losses: List["torch.Tensor"]) -> "torch.Tensor":
        total = losses[0].new_tensor(0.0)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total

    def sigmas(self) -> List[float]:
        with torch.no_grad():
            return torch.exp(0.5 * self.log_vars.detach()).cpu().tolist()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _topk_accuracy(logits: "torch.Tensor", y: "torch.Tensor", k: int) -> float:
    with torch.no_grad():
        k = min(k, logits.shape[1])
        topk = logits.topk(k, dim=1).indices
        hit = (topk == y.unsqueeze(1)).any(dim=1).float().mean()
        return float(hit.item())


def _counts_from_samples(samples, field: str, num_classes: int) -> "torch.Tensor":
    values = [int(getattr(s, field)) for s in samples]
    t = torch.as_tensor(values, dtype=torch.long)
    counts = torch.bincount(t, minlength=num_classes).float()
    return counts


def _class_weights_from_counts(counts: "torch.Tensor") -> "torch.Tensor":
    # inverse-sqrt frequency; stable for long-tail artist labels.
    w = 1.0 / torch.sqrt(torch.clamp(counts, min=1.0))
    w = w / w.mean().clamp(min=1e-8)
    return w.float()


def _build_sample_weights(
    samples,
    style_counts: "torch.Tensor",
    artist_counts: "torch.Tensor",
    genre_counts: "torch.Tensor",
) -> "torch.Tensor":
    sw = 1.0 / torch.clamp(style_counts, min=1.0)
    aw = 1.0 / torch.clamp(artist_counts, min=1.0)
    gw = 1.0 / torch.clamp(genre_counts, min=1.0)

    weights = []
    for s in samples:
        w = (
            float(sw[int(s.style_label)])
            + float(aw[int(s.artist_label)])
            + float(gw[int(s.genre_label)])
        ) / 3.0
        weights.append(w)

    out = torch.as_tensor(weights, dtype=torch.double)
    out = out / out.mean().clamp(min=1e-12)
    return out


def _run_epoch(
    model: "CRNNMultiTask",
    loader: "DataLoader",
    device: "torch.device",
    criterion_style: "nn.Module",
    criterion_artist: "nn.Module",
    criterion_genre: "nn.Module",
    optimizer: "torch.optim.Optimizer | None" = None,
    mtloss: "MultiTaskUncertaintyLoss | None" = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    n = 0
    loss_sum = 0.0
    loss_style_sum = 0.0
    loss_artist_sum = 0.0
    loss_genre_sum = 0.0
    style_top1 = 0.0
    artist_top1 = 0.0
    artist_top5 = 0.0
    genre_top1 = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        ys = labels["style"].to(device, non_blocking=True)
        ya = labels["artist"].to(device, non_blocking=True)
        yg = labels["genre"].to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        out = model(images)
        ls = criterion_style(out["style"], ys)
        la = criterion_artist(out["artist"], ya)
        lg = criterion_genre(out["genre"], yg)

        if mtloss is None:
            loss = ls + la + lg
        else:
            loss = mtloss([ls, la, lg])

        if train_mode:
            loss.backward()
            optimizer.step()

        b = images.size(0)
        n += b
        loss_sum += float(loss.item()) * b
        loss_style_sum += float(ls.item()) * b
        loss_artist_sum += float(la.item()) * b
        loss_genre_sum += float(lg.item()) * b
        style_top1 += float((out["style"].argmax(1) == ys).float().sum().item())
        artist_top1 += float((out["artist"].argmax(1) == ya).float().sum().item())
        artist_top5 += _topk_accuracy(out["artist"], ya, 5) * b
        genre_top1 += float((out["genre"].argmax(1) == yg).float().sum().item())
        if n > 40:
            break

    if n == 0:
        return {
            "loss": 0.0,
            "loss_style": 0.0,
            "loss_artist": 0.0,
            "loss_genre": 0.0,
            "style_top1": 0.0,
            "artist_top1": 0.0,
            "artist_top5": 0.0,
            "genre_top1": 0.0,
        }

    return {
        "loss": loss_sum / n,
        "loss_style": loss_style_sum / n,
        "loss_artist": loss_artist_sum / n,
        "loss_genre": loss_genre_sum / n,
        "style_top1": style_top1 / n,
        "artist_top1": artist_top1 / n,
        "artist_top5": artist_top5 / n,
        "genre_top1": genre_top1 / n,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="CRNN multi-task training entrypoint")
    p.add_argument("--config", default="configs/baseline.json")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default="outputs/train")
    args = p.parse_args()

    if _IMPORT_ERROR is not None:
        print(f"error: torch/torchvision unavailable: {_IMPORT_ERROR}", file=sys.stderr)
        return 2

    from artextract.models import CRNNMultiTask

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    train_manifest = data_cfg.get("train_manifest")
    val_manifest = data_cfg.get("val_manifest")
    images_root = data_cfg.get("images_root")
    metadata_dir = data_cfg.get("metadata_dir")
    if not train_manifest or not val_manifest or not images_root:
        print("error: config.data must define train_manifest, val_manifest, images_root")
        return 2

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(seed)

    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 20))
    batch_size = (
        args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 32))
    )
    lr = args.lr if args.lr is not None else float(train_cfg.get("lr", 3e-4))
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(train_cfg.get("num_workers", 4))
    )

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    image_size = int(model_cfg.get("image_size", 224))
    use_augment = bool(train_cfg.get("augment", True))
    if use_augment:
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
    transform_val = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_ds = MultiTaskImageDataset(train_manifest, images_root, transform=transform_train)
    val_ds = MultiTaskImageDataset(val_manifest, images_root, transform=transform_val)

    if metadata_dir:
        style_classes = len(load_class_map(Path(metadata_dir) / "style_class.txt"))
        artist_classes = len(load_class_map(Path(metadata_dir) / "artist_class.txt"))
        genre_classes = len(load_class_map(Path(metadata_dir) / "genre_class.txt"))
    else:
        style_classes = max(s.style_label for s in train_ds.samples) + 1
        artist_classes = max(s.artist_label for s in train_ds.samples) + 1
        genre_classes = max(s.genre_label for s in train_ds.samples) + 1

    style_counts = _counts_from_samples(train_ds.samples, "style_label", style_classes)
    artist_counts = _counts_from_samples(train_ds.samples, "artist_label", artist_classes)
    genre_counts = _counts_from_samples(train_ds.samples, "genre_label", genre_classes)

    use_weighted_sampler = bool(train_cfg.get("weighted_sampler", True))
    if use_weighted_sampler:
        sample_weights = _build_sample_weights(train_ds.samples, style_counts, artist_counts, genre_counts)
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = CRNNMultiTask(
        style_classes=style_classes,
        artist_classes=artist_classes,
        genre_classes=genre_classes,
        patch_grid=int(model_cfg.get("patch_grid", 4)),
        cnn_backbone=str(model_cfg.get("cnn_backbone", "resnet18")),
        pretrained_backbone=bool(model_cfg.get("pretrained_backbone", False)),
        global_dim=int(model_cfg.get("global_dim", 512)),
        patch_dim=int(model_cfg.get("patch_dim", 256)),
        rnn_hidden=int(model_cfg.get("rnn_hidden", 256)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)

    use_class_balanced_loss = bool(train_cfg.get("class_balanced_loss", True))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    style_w = _class_weights_from_counts(style_counts).to(device) if use_class_balanced_loss else None
    artist_w = _class_weights_from_counts(artist_counts).to(device) if use_class_balanced_loss else None
    genre_w = _class_weights_from_counts(genre_counts).to(device) if use_class_balanced_loss else None

    criterion_style = nn.CrossEntropyLoss(weight=style_w, label_smoothing=label_smoothing)
    criterion_artist = nn.CrossEntropyLoss(weight=artist_w, label_smoothing=label_smoothing)
    criterion_genre = nn.CrossEntropyLoss(weight=genre_w, label_smoothing=label_smoothing)

    use_uncertainty_loss = bool(train_cfg.get("multitask_uncertainty", True))
    mtloss = MultiTaskUncertaintyLoss(num_tasks=3).to(device) if use_uncertainty_loss else None

    opt_params: Iterable["torch.nn.Parameter"]
    if mtloss is not None:
        opt_params = list(model.parameters()) + list(mtloss.parameters())
    else:
        opt_params = model.parameters()

    optimizer = torch.optim.AdamW(
        opt_params,
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    out_dir = Path(args.out_dir)
    ensure_parent_dir(out_dir / "x")
    history = []
    best_style = -1.0

    for epoch in range(1, epochs + 1):
        train_m = _run_epoch(
            model,
            train_loader,
            device,
            criterion_style=criterion_style,
            criterion_artist=criterion_artist,
            criterion_genre=criterion_genre,
            optimizer=optimizer,
            mtloss=mtloss,
        )
        with torch.no_grad():
            val_m = _run_epoch(
                model,
                val_loader,
                device,
                criterion_style=criterion_style,
                criterion_artist=criterion_artist,
                criterion_genre=criterion_genre,
                optimizer=None,
                mtloss=mtloss,
            )
        scheduler.step()

        row = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"])}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}": v for k, v in val_m.items()})
        if mtloss is not None:
            s = mtloss.sigmas()
            row["sigma_style"] = float(s[0])
            row["sigma_artist"] = float(s[1])
            row["sigma_genre"] = float(s[2])
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_m['loss']:.4f} val_loss={val_m['loss']:.4f} "
            f"val_style_top1={val_m['style_top1']:.4f} val_artist_top5={val_m['artist_top5']:.4f} "
            f"val_genre_top1={val_m['genre_top1']:.4f}"
        )

        ckpt_last = out_dir / "last_model.pt"
        torch.save(model.state_dict(), ckpt_last)
        if val_m["style_top1"] > best_style:
            best_style = val_m["style_top1"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    hist_path = out_dir / "training_history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    run_meta = {
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "augment": use_augment,
        "weighted_sampler": use_weighted_sampler,
        "class_balanced_loss": use_class_balanced_loss,
        "label_smoothing": label_smoothing,
        "multitask_uncertainty": use_uncertainty_loss,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"done: best_val_style_top1={best_style:.4f}")
    print(f"artifacts: {out_dir / 'best_model.pt'} {out_dir / 'last_model.pt'} {hist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
