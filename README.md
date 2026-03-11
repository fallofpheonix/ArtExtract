# ArtExtract

Unified repository for both required tasks:
1. Task 1: multi-task painting classification (style/artist/genre).
2. Task 2: hidden-art retrieval baseline (synthetic overpaint + reconstruction model).

## Scope
This repo contains code, configs, metadata, docs, notebooks, and reports.
Large binaries and runtime artifacts (raw images, checkpoints, outputs, virtualenv, zip archives) are intentionally excluded from version control.

## Repository Structure
```text
ArtExtract/
  configs/                 # static baseline configs
  data/
    manifests/             # generated/seed manifests
    metadata/wikiart_csv/  # split metadata + label maps
    templates/             # templates/helpers
  docs/                    # operational documentation
  notebooks/               # executed analysis notebooks and exports
  reports/                 # run summaries
  scripts/                 # runnable entrypoints (task1/task2)
  src/artextract/          # package code (data/models/training/retrieval/utils)
  tests/                   # test notes/placeholders
  DATA_RIGHTS.md
  requirements.txt
```

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Task 1 (classification pipeline):
```bash
bash scripts/run_quick_pipeline.sh
```

Task 2 (hidden retrieval pipeline):
```bash
bash scripts/run_hidden_retrieval_pipeline.sh
```

## Task 1 Controls
```bash
# quick mode (default)
QUICK_MODE=1 PER_STYLE_TRAIN=30 PER_STYLE_VAL=10 EPOCHS=3 BATCH_SIZE=16 bash scripts/run_quick_pipeline.sh

# larger quick subset
QUICK_MODE=1 PER_STYLE_TRAIN=120 PER_STYLE_VAL=40 EPOCHS=10 bash scripts/run_quick_pipeline.sh

# full manifest mode
QUICK_MODE=0 MAX_IMAGES=30000 EPOCHS=10 BATCH_SIZE=16 bash scripts/run_quick_pipeline.sh
```

Task 1 outputs (under `OUT_DIR`, default `outputs/quick_cpu`):
1. `best_model.pt`
2. `last_model.pt`
3. `training_history.json`
4. `val_metrics.json`
5. `val_predictions.csv`
6. `val_embeddings.npy`
7. `val_style_labels.npy`
8. `style_outliers.csv`

## Task 2 Controls
```bash
EPOCHS=3 BATCH_SIZE=12 MAX_IMAGES=2500 OUT_DIR=outputs/hidden_retrieval_quick bash scripts/run_hidden_retrieval_pipeline.sh
```

Task 2 outputs (under `OUT_DIR`):
1. `best_model.pt`
2. `metrics.json`
3. `val_metrics_eval.json`
4. `val_preview.png`

## Data Expectations
1. Runners auto-detect local image roots (`data/raw/wikiart` preferred, fallback `data/raw/images`).
2. If `wikiart.zip` is present locally, runner can extract into `data/raw/`.
3. Manifests are filtered to existing/readable images before training.

## Documentation
1. [docs/README.md](docs/README.md)
2. [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
3. [docs/TRAINING_AND_EVAL.md](docs/TRAINING_AND_EVAL.md)
4. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
5. [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)
6. [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
7. [docs/DATA_AND_COMPLIANCE.md](docs/DATA_AND_COMPLIANCE.md)

## Compliance
Read [DATA_RIGHTS.md](DATA_RIGHTS.md) before distributing data or model artifacts.
