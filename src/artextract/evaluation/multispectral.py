from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from artextract.data import MultiSpectralDataset, multispectral_collate
from artextract.evaluation.metrics import (
    auc_trapezoid,
    binary_accuracy,
    binary_confusion,
    binary_precision_recall_f1,
    mean,
    multi_label_f1,
    psnr,
    roc_curve_binary,
    ssim_simple,
)
from artextract.models import MultiSpectralMultiTaskModel

try:
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
except Exception as e:  # pragma: no cover
    plt = None
    torch = None
    DataLoader = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _resolve_device(requested: str | None) -> "torch.device":
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_roc(fpr: np.ndarray, tpr: np.ndarray, auc_v: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC={auc_v:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Hidden Image ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_recon_examples(
    xs: np.ndarray,
    preds: np.ndarray,
    gts: np.ndarray,
    out_dir: Path,
    limit: int = 5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(limit, xs.shape[0])

    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(xs[i][0], cmap="gray")
        axes[0].set_title("Input (ch0)")
        axes[1].imshow(preds[i][0], cmap="gray")
        axes[1].set_title("Pred")
        axes[2].imshow(gts[i][0], cmap="gray")
        axes[2].set_title("Target")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"example_{i:03d}.png", dpi=140)
        plt.close(fig)


def evaluate_multispectral(
    manifest_path: str | Path,
    channels: Sequence[str],
    tasks: Iterable[str],
    cfg: Dict[str, object],
    checkpoint_path: str | Path,
    out_dir: str | Path,
    pigments_vocab_path: str | Path,
    device: str | None = None,
) -> Path:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"evaluation deps unavailable: {_IMPORT_ERROR}")

    tasks_set = {t.strip().lower() for t in tasks}
    if not tasks_set:
        raise ValueError("tasks cannot be empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(pigments_vocab_path).open("r", encoding="utf-8") as f:
        pigments_vocab = json.load(f)

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}

    image_size = int(model_cfg.get("image_size", 128))

    val_ds = MultiSpectralDataset(
        manifest_path=manifest_path,
        channels_order=channels,
        split="val",
        tasks=tasks_set,
        pigments_vocab=pigments_vocab,
        image_size=image_size,
        strict_dimensions=bool(model_cfg.get("strict_dimensions", False)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=False,
        collate_fn=multispectral_collate,
    )

    dev = _resolve_device(device or train_cfg.get("device"))
    model = MultiSpectralMultiTaskModel(
        in_channels=len(channels),
        num_pigments=len(pigments_vocab),
        enable_properties=("properties" in tasks_set),
        enable_hidden=("hidden" in tasks_set),
        enable_reconstruction=("reconstruction" in tasks_set),
        base_channels=int(model_cfg.get("base_channels", 32)),
    ).to(dev)

    state = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()

    all_hidden_true = []
    all_hidden_pred = []
    all_hidden_prob = []
    all_damage_true = []
    all_damage_pred = []
    all_rest_true = []
    all_rest_pred = []
    all_pig_true = []
    all_pig_pred = []

    recon_psnr = []
    recon_ssim = []
    recon_x = []
    recon_p = []
    recon_t = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(dev, non_blocking=True)
            mask = batch["channel_mask"].to(dev, non_blocking=True)
            targets = {k: v.to(dev, non_blocking=True) for k, v in batch["targets"].items()}

            out = model(x, mask)

            if "hidden" in tasks_set:
                prob = torch.sigmoid(out["hidden_logits"]).cpu().numpy()
                pred = (prob >= 0.5).astype(np.int64)
                true = targets["hidden_image"].cpu().numpy().astype(np.int64)
                all_hidden_prob.extend(prob.tolist())
                all_hidden_pred.extend(pred.tolist())
                all_hidden_true.extend(true.tolist())

            if "properties" in tasks_set:
                pig_prob = torch.sigmoid(out["pigments_logits"]).cpu().numpy()
                pig_pred = (pig_prob >= 0.5).astype(np.int64)
                pig_true = targets["pigments"].cpu().numpy().astype(np.int64)
                all_pig_pred.append(pig_pred)
                all_pig_true.append(pig_true)

                dam_prob = torch.sigmoid(out["damage_logits"]).cpu().numpy()
                dam_pred = (dam_prob >= 0.5).astype(np.int64)
                dam_true = targets["damage"].cpu().numpy().astype(np.int64)
                all_damage_pred.extend(dam_pred.tolist())
                all_damage_true.extend(dam_true.tolist())

                res_prob = torch.sigmoid(out["restoration_logits"]).cpu().numpy()
                res_pred = (res_prob >= 0.5).astype(np.int64)
                res_true = targets["restoration"].cpu().numpy().astype(np.int64)
                all_rest_pred.extend(res_pred.tolist())
                all_rest_true.extend(res_true.tolist())

            if "reconstruction" in tasks_set:
                recon_pred = out["reconstruction"].cpu().numpy()
                recon_target = targets["hidden_gt"].cpu().numpy()
                inp = x.cpu().numpy()

                mse_vals = ((recon_pred - recon_target) ** 2).mean(axis=(1, 2, 3))
                for i in range(recon_pred.shape[0]):
                    recon_psnr.append(psnr(float(mse_vals[i])))
                    recon_ssim.append(ssim_simple(recon_pred[i], recon_target[i]))

                recon_x.append(inp)
                recon_p.append(recon_pred)
                recon_t.append(recon_target)

    metrics: Dict[str, object] = {
        "samples_val": len(val_ds),
    }

    if "properties" in tasks_set:
        pig_true = np.concatenate(all_pig_true, axis=0) if all_pig_true else np.zeros((0, len(pigments_vocab)))
        pig_pred = np.concatenate(all_pig_pred, axis=0) if all_pig_pred else np.zeros((0, len(pigments_vocab)))

        damage_true = np.asarray(all_damage_true, dtype=np.int64)
        damage_pred = np.asarray(all_damage_pred, dtype=np.int64)
        rest_true = np.asarray(all_rest_true, dtype=np.int64)
        rest_pred = np.asarray(all_rest_pred, dtype=np.int64)

        d_prec, d_rec, d_f1 = binary_precision_recall_f1(damage_true, damage_pred)
        r_prec, r_rec, r_f1 = binary_precision_recall_f1(rest_true, rest_pred)

        metrics["properties"] = {
            "pigments_macro_f1": multi_label_f1(pig_true, pig_pred),
            "damage_accuracy": binary_accuracy(damage_true, damage_pred),
            "damage_precision": d_prec,
            "damage_recall": d_rec,
            "damage_f1": d_f1,
            "restoration_accuracy": binary_accuracy(rest_true, rest_pred),
            "restoration_precision": r_prec,
            "restoration_recall": r_rec,
            "restoration_f1": r_f1,
        }

    if "hidden" in tasks_set:
        y_true = np.asarray(all_hidden_true, dtype=np.int64)
        y_pred = np.asarray(all_hidden_pred, dtype=np.int64)
        y_prob = np.asarray(all_hidden_prob, dtype=np.float32)

        p, r, f1 = binary_precision_recall_f1(y_true, y_pred)
        cm = binary_confusion(y_true, y_pred)
        fpr, tpr, _ = roc_curve_binary(y_true, y_prob)
        auc_v = auc_trapezoid(fpr, tpr)

        metrics["hidden"] = {
            "accuracy": binary_accuracy(y_true, y_pred),
            "precision": p,
            "recall": r,
            "f1": f1,
            "auc": auc_v,
        }

        _save_confusion_matrix(cm, out_dir / "confusion_matrix.png", title="Hidden Image Confusion Matrix")
        _save_roc(fpr, tpr, auc_v, out_dir / "roc_curve.png")

    if "reconstruction" in tasks_set:
        metrics["reconstruction"] = {
            "psnr": mean(recon_psnr),
            "ssim": mean(recon_ssim),
        }
        if recon_x:
            x = np.concatenate(recon_x, axis=0)
            p = np.concatenate(recon_p, axis=0)
            t = np.concatenate(recon_t, axis=0)
            _save_recon_examples(x, p, t, out_dir / "recon_examples", limit=5)

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    run_meta = {
        "channels": list(channels),
        "tasks": sorted(tasks_set),
        "manifest_path": str(Path(manifest_path).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
    }
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    return metrics_path
