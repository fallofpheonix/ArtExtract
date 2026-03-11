# CLI Reference

## `scripts/run_quick_pipeline.sh`
One-command end-to-end quick run.

Environment variables:
1. `VENV_DIR` (default `.venv`)
2. `EPOCHS` (default `3`)
3. `BATCH_SIZE` (default `16`)
4. `MAX_IMAGES` (default `0` = all from quick manifests)
5. `WORKERS` (default `16`)
6. `OUT_DIR` (default `outputs/quick_cpu`)
7. `QUICK_MODE` (default `1`; `0` uses full manifests)
8. `PER_STYLE_TRAIN` (default `30`; quick mode only)
9. `PER_STYLE_VAL` (default `10`; quick mode only)
10. `SKIP_DOWNLOAD` (default `0`; `1` skips downloader stage)
11. `DOWNLOAD_TIMEOUT` (default `8`; downloader URL timeout)
12. `TRAIN_WORKERS` (default `0`; DataLoader workers)
13. `IMAGES_ROOT` (default auto-detect between `data/raw/wikiart` and `data/raw/images`)
14. `VALIDATE_IMAGES` (default `1`; decode-check and drop unreadable rows)
15. `RUNTIME_CONFIG` (default `configs/.runtime_quick_cpu.json`; auto-generated)

## `scripts/build_manifest.py`
```bash
python scripts/build_manifest.py --split {train|val} --out <csv> [--metadata-dir data/metadata/wikiart_csv]
```

## `scripts/dataset_stats.py`
```bash
python scripts/dataset_stats.py --manifest <csv>
```

## `scripts/download_images_from_manifest.py`
```bash
python scripts/download_images_from_manifest.py \
  --manifest <csv> [--manifest <csv> ...] \
  [--out-root data/raw/images] [--max-images 0] [--workers 16] [--timeout 15]
```

## `scripts/train_crnn.py`
```bash
python scripts/train_crnn.py \
  [--config configs/baseline.json] \
  [--epochs N] [--batch-size N] [--lr F] [--num-workers N] [--device cpu|cuda] \
  [--out-dir outputs/train]
```

## `scripts/evaluate_model.py`
```bash
python scripts/evaluate_model.py \
  [--config configs/baseline.json] \
  --checkpoint <model.pt> \
  [--out outputs/val_metrics.json] \
  [--predictions-out outputs/val_predictions.csv] \
  [--embeddings-out outputs/val_embeddings.npy] \
  [--style-labels-out outputs/val_style_labels.npy]
```

## `scripts/detect_outliers.py`
```bash
python scripts/detect_outliers.py \
  --embeddings <npy> --labels <npy> --out <csv> \
  [--contamination 0.02] [--min-samples-per-class 20]
```

## `scripts/evaluate_predictions.py`
```bash
python scripts/evaluate_predictions.py --predictions <csv(y_true,y_pred)>
```

## `scripts/run_hidden_retrieval_pipeline.sh`
One-command optional hidden-image retrieval baseline.

Environment variables:
1. `VENV_DIR` (default `.venv`)
2. `EPOCHS` (default `3`)
3. `BATCH_SIZE` (default `12`)
4. `MAX_IMAGES` (default `2500`)
5. `OUT_DIR` (default `outputs/hidden_retrieval_quick`)
6. `TRAIN_WORKERS` (default `0`)
7. `IMAGES_ROOT` (optional; default auto-detect)
8. `CONFIG_PATH` (default `configs/retrieval_baseline.json`)
9. `RUNTIME_CONFIG` (default `configs/.runtime_retrieval.json`)

## `scripts/train_hidden_retrieval.py`
```bash
python scripts/train_hidden_retrieval.py \
  [--config configs/retrieval_baseline.json] \
  [--epochs N] [--batch-size N] [--lr F] [--num-workers N] \
  [--device cpu|cuda|mps] [--out-dir outputs/hidden_retrieval]
```

## `scripts/evaluate_hidden_retrieval.py`
```bash
python scripts/evaluate_hidden_retrieval.py \
  [--config configs/retrieval_baseline.json] \
  --checkpoint outputs/hidden_retrieval/best_model.pt \
  [--out outputs/hidden_retrieval/val_metrics_eval.json]
```

## `scripts/filter_redistributable.py`
```bash
python scripts/filter_redistributable.py \
  --input <provenance.csv> \
  --out-allowed <csv> \
  --out-blocked <csv>
```
