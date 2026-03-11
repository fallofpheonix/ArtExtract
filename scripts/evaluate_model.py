#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artextract.config import load_config
from artextract.data import MultiTaskImageDataset, load_class_map
from artextract.models import CRNNMultiTask
from artextract.training.metrics import classification_metrics


def main() -> int:
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except Exception as e:
        print(f"error: torch/torchvision unavailable: {e}", file=sys.stderr)
        return 2

    p = argparse.ArgumentParser(description="Evaluate trained CRNN checkpoint on val manifest")
    p.add_argument("--config", default="configs/baseline.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="outputs/val_metrics.json")
    p.add_argument("--predictions-out", default=None, help="Optional CSV export of per-sample predictions")
    p.add_argument("--embeddings-out", default=None, help="Optional .npy export for fused embeddings")
    p.add_argument("--style-labels-out", default=None, help="Optional .npy export for style labels")
    args = p.parse_args()

    cfg = load_config(args.config).payload
    dcfg = cfg["data"]
    mcfg = cfg["model"]

    transform = transforms.Compose(
        [
            transforms.Resize((int(mcfg.get("image_size", 224)), int(mcfg.get("image_size", 224)))),
            transforms.ToTensor(),
        ]
    )
    val_ds = MultiTaskImageDataset(dcfg["val_manifest"], dcfg["images_root"], transform=transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    md = Path(dcfg["metadata_dir"])
    style_n = len(load_class_map(md / "style_class.txt"))
    artist_n = len(load_class_map(md / "artist_class.txt"))
    genre_n = len(load_class_map(md / "genre_class.txt"))

    model = CRNNMultiTask(
        style_classes=style_n,
        artist_classes=artist_n,
        genre_classes=genre_n,
        patch_grid=int(mcfg.get("patch_grid", 4)),
        cnn_backbone=str(mcfg.get("cnn_backbone", "resnet18")),
        pretrained_backbone=bool(mcfg.get("pretrained_backbone", False)),
        global_dim=int(mcfg.get("global_dim", 512)),
        patch_dim=int(mcfg.get("patch_dim", 256)),
        rnn_hidden=int(mcfg.get("rnn_hidden", 256)),
        dropout=float(mcfg.get("dropout", 0.2)),
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    ys, ps = [], []
    ya, pa = [], []
    yg, pg = [], []
    relpaths = []
    emb_chunks = []
    top5_hits = 0
    n = 0

    with torch.no_grad():
        offset = 0
        for images, labels in val_loader:
            out = model(images)
            ts = labels["style"]
            ta = labels["artist"]
            tg = labels["genre"]

            psb = out["style"].argmax(1)
            pab = out["artist"].argmax(1)
            pgb = out["genre"].argmax(1)

            top5 = out["artist"].topk(min(5, out["artist"].shape[1]), dim=1).indices
            top5_hits += int((top5 == ta.unsqueeze(1)).any(dim=1).sum().item())
            n += int(ta.shape[0])

            ys.extend(ts.tolist())
            ps.extend(psb.tolist())
            ya.extend(ta.tolist())
            pa.extend(pab.tolist())
            yg.extend(tg.tolist())
            pg.extend(pgb.tolist())

            emb_chunks.append(out["embedding"].detach().cpu())
            b = int(ta.shape[0])
            for i in range(offset, offset + b):
                relpaths.append(val_ds.samples[i].image_relpath)
            offset += b

    metrics = {
        "style": classification_metrics(ys, ps),
        "artist": classification_metrics(ya, pa),
        "genre": classification_metrics(yg, pg),
        "artist_top5": float(top5_hits / n if n else 0.0),
        "samples_val": n,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.predictions_out:
        p_out = Path(args.predictions_out)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        with p_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "image_relpath",
                    "style_true",
                    "style_pred",
                    "artist_true",
                    "artist_pred",
                    "genre_true",
                    "genre_pred",
                ]
            )
            for i in range(n):
                w.writerow([relpaths[i], ys[i], ps[i], ya[i], pa[i], yg[i], pg[i]])

    if args.embeddings_out:
        import numpy as np

        e_out = Path(args.embeddings_out)
        e_out.parent.mkdir(parents=True, exist_ok=True)
        emb = torch.cat(emb_chunks, dim=0).numpy() if emb_chunks else np.zeros((0, 0), dtype=np.float32)
        np.save(e_out, emb)

    if args.style_labels_out:
        import numpy as np

        l_out = Path(args.style_labels_out)
        l_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(l_out, np.asarray(ys, dtype=np.int64))

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
