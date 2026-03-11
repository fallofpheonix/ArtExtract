# ArtExtract

Consolidated HumanAI repository for **Painting in a Painting**:
1. Multispectral property prediction (pigments, damage, restoration)
2. Hidden-image detection
3. Optional hidden-image reconstruction
4. Task2 similarity retrieval track (CLIP + FAISS)

Legacy RGB baselines are preserved for backward compatibility.

## Repository Layout
```text
ArtExtract/
  src/artextract/
    data/             # manifests + loaders + synthetic multispectral harness
    models/           # CRNN + multispectral multitask model
    training/         # train engines
    evaluation/       # eval engines + metrics/artifacts
    reconstruction/   # U-Net decoder modules
    similarity/       # CLIP/FAISS similarity pipeline
  scripts/
    train.py
    eval.py
    run_similarity.py
    run_reconstruction.py
    generate_synthetic_multispectral.py
    run_quick_pipeline.sh               # legacy task1 baseline
    run_hidden_retrieval_pipeline.sh    # legacy hidden retrieval baseline
  configs/
    multispectral_baseline.json
  notebooks/
    task1/
    task2/
  docs/
  reports/
  data/
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional pinned lock:
pip install -r requirements.lock.txt
```

## Unified Multispectral Pipeline
Train (manifest-driven):
```bash
python scripts/train.py \
  --manifest data/manifests/multispectral.csv \
  --channels rgb,ir,uv,xray \
  --tasks properties,hidden,reconstruction \
  --config configs/multispectral_baseline.json \
  --out-dir reports/run_ms
```

Evaluate:
```bash
python scripts/eval.py \
  --manifest reports/run_ms/resolved_manifest.csv \
  --checkpoint reports/run_ms/best_model.pt \
  --pigments-vocab reports/run_ms/pigments_vocab.json \
  --channels rgb,ir,uv,xray \
  --tasks properties,hidden,reconstruction \
  --config configs/multispectral_baseline.json \
  --out-dir reports/run_ms
```

## Synthetic Harness (No Real Multispectral Data)
Generate synthetic multispectral data from local RGB paintings:
```bash
python scripts/generate_synthetic_multispectral.py \
  --images-root /path/to/images \
  --out-root data/synthetic_multispectral \
  --channels rgb,ir,uv,xray \
  --max-samples 500
```

One-command hidden+reconstruction smoke run:
```bash
python scripts/run_reconstruction.py --images-root /path/to/images
```

## Similarity Track
```bash
python scripts/run_similarity.py --images-dir images --opendata-dir nga_data
```
(Requires optional similarity dependencies in `requirements_similarity.txt`.)

## HumanAI Evaluation Artifacts
1. Task1 evidence: `https://github.com/fallofpheonix/ArtExtract/tree/main/test_results/task1`
2. Task2 evidence: `https://github.com/fallofpheonix/ArtExtract/tree/main/test_results/task2`

Task2 validation note:
1. Pipeline is validated through dependency/CLI/runtime startup stages.
2. Current runtime log shows failure at remote model fetch stage due Hugging Face network timeout.
3. This is an external fetch issue, not a local preprocessing/schema/CLI failure.

## Standard Report Artifacts
Each unified run writes:
1. `reports/<run_id>/metrics.json`
2. `reports/<run_id>/run_meta.json`
3. `reports/<run_id>/confusion_matrix.png` (when hidden task enabled)
4. `reports/<run_id>/roc_curve.png` (when hidden task enabled)
5. `reports/<run_id>/recon_examples/*` (when reconstruction enabled)

## Legacy Baselines (Preserved)
1. `bash scripts/run_quick_pipeline.sh`
2. `bash scripts/run_hidden_retrieval_pipeline.sh`

## Compliance
Read [DATA_RIGHTS.md](DATA_RIGHTS.md) before redistributing data/model artifacts.
