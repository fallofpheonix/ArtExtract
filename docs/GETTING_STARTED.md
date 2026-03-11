# Getting Started

## Prerequisites
1. macOS/Linux shell.
2. Python 3.10+.
3. Network access (for WikiArt image download).

## Quick Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## One-Command Pipeline
```bash
bash scripts/run_quick_pipeline.sh
```

This performs:
1. Environment bootstrap.
2. Local dataset detection (`wikiart.zip` / `data/raw/wikiart` / `data/raw/images`).
3. Manifest generation.
4. Quick subset creation (or full-manifest mode).
5. Optional image download.
6. Filtering manifests to existing/readable images.
7. Training.
8. Evaluation.
9. Outlier detection.

## Optional Hidden Retrieval (One Command)
```bash
bash scripts/run_hidden_retrieval_pipeline.sh
```

Quick overrides:
```bash
EPOCHS=2 MAX_IMAGES=1200 BATCH_SIZE=10 OUT_DIR=outputs/hidden_retrieval_quick bash scripts/run_hidden_retrieval_pipeline.sh
```

## Optional Runtime Overrides
```bash
EPOCHS=5 BATCH_SIZE=16 MAX_IMAGES=500 WORKERS=16 OUT_DIR=outputs/quick_cpu bash scripts/run_quick_pipeline.sh
```

Variables:
1. `EPOCHS` (default `3`)
2. `BATCH_SIZE` (default `16`)
3. `MAX_IMAGES` (default `0`, meaning all in quick manifests)
4. `WORKERS` (default `16` for downloader)
5. `OUT_DIR` (default `outputs/quick_cpu`)
6. `VENV_DIR` (default `.venv`)
7. `QUICK_MODE` (`1` quick subset, `0` full-manifest mode)
8. `PER_STYLE_TRAIN` (default `30`, only used when `QUICK_MODE=1`)
9. `PER_STYLE_VAL` (default `10`, only used when `QUICK_MODE=1`)
10. `SKIP_DOWNLOAD` (`1` to skip downloader and use cached images)
11. `DOWNLOAD_TIMEOUT` (seconds per URL attempt in downloader)
12. `TRAIN_WORKERS` (default `0`; DataLoader workers for training/eval)
13. `IMAGES_ROOT` (optional; force image root instead of auto-detect)
14. `VALIDATE_IMAGES` (default `1`; drops unreadable images)
15. `RUNTIME_CONFIG` (default `configs/.runtime_quick_cpu.json`)

## Expected Artifacts
After success:
1. `outputs/quick_cpu/best_model.pt`
2. `outputs/quick_cpu/training_history.json`
3. `outputs/quick_cpu/val_metrics.json`
4. `outputs/quick_cpu/val_predictions.csv`
5. `outputs/quick_cpu/style_outliers.csv`

## Validate Run
```bash
python -m json.tool outputs/quick_cpu/val_metrics.json
```
