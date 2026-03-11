# Training and Evaluation

## 1. Build Multitask Manifests
```bash
python scripts/build_manifest.py --split train --out data/manifests/train_multitask.csv
python scripts/build_manifest.py --split val --out data/manifests/val_multitask.csv
```

Behavior:
1. Reads `style_{split}.csv`, `artist_{split}.csv`, `genre_{split}.csv`.
2. Joins on `image_relpath`.
3. Drops rows missing artist or genre match.

## 2. Train
```bash
python scripts/train_crnn.py \
  --config configs/quick_cpu.json \
  --epochs 3 \
  --batch-size 16 \
  --out-dir outputs/quick_cpu
```

Train outputs:
1. `best_model.pt` (best style val top-1)
2. `last_model.pt`
3. `training_history.json`

## 3. Evaluate Checkpoint
```bash
python scripts/evaluate_model.py \
  --config configs/quick_cpu.json \
  --checkpoint outputs/quick_cpu/best_model.pt \
  --out outputs/quick_cpu/val_metrics.json \
  --predictions-out outputs/quick_cpu/val_predictions.csv \
  --embeddings-out outputs/quick_cpu/val_embeddings.npy \
  --style-labels-out outputs/quick_cpu/val_style_labels.npy
```

Metrics written:
1. Style: top-1, macro/weighted F1, balanced accuracy.
2. Artist: top-1, top-5, macro/weighted F1, balanced accuracy.
3. Genre: top-1, macro/weighted F1, balanced accuracy.

## 4. Detect Outliers
```bash
python scripts/detect_outliers.py \
  --embeddings outputs/quick_cpu/val_embeddings.npy \
  --labels outputs/quick_cpu/val_style_labels.npy \
  --out outputs/quick_cpu/style_outliers.csv \
  --contamination 0.1 \
  --min-samples-per-class 5
```

## 5. Report Inputs
Recommended minimum files for reporting:
1. `training_history.json`
2. `val_metrics.json`
3. `val_predictions.csv`
4. `style_outliers.csv`

## 6. Optional Hidden-Image Retrieval
Baseline pipeline:
1. Synthetic overpaint data generation from WikiArt images.
2. U-Net reconstruction model.
3. Metrics: MAE, MSE, PSNR.

Entrypoints:
1. `scripts/run_hidden_retrieval_pipeline.sh`
2. `scripts/train_hidden_retrieval.py`
3. `scripts/evaluate_hidden_retrieval.py`

## Known Limits of Current Pipeline
1. Loss is unweighted sum of three cross-entropy heads.
2. Default setup uses lightweight augmentation; no advanced policy search (RandAugment/CutMix/MixUp) is enabled.
3. Manifest intersection currently reduces class coverage versus full raw CSV cardinality.
4. `build_manifest.py` keeps only intersection across style/artist/genre splits (`train: 11276`, `val: 4707` with current metadata).
