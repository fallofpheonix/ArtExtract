# Hidden Retrieval Optional

## Goal
Retrieve an underpainting estimate from a visible painting image using a synthetic training setup.

## Data Flow
1. Sample two paintings: cover `C`, hidden `H`.
2. Generate random soft mask `M` and opacity `a`.
3. Synthesize observed image: `O = C*(1-aM) + H*(aM)`.
4. Train model to map `O -> H`.

## Model
- `UNetRetrieval` (`src/artextract/retrieval/model.py`)
- Input: 3-channel observed image.
- Output: 3-channel reconstructed hidden image.

## Loss
- `L = 1.0 * L1 + 0.5 * MSE` (configurable).

## Run
- One command: `scripts/run_hidden_retrieval_pipeline.sh`
- Config: `configs/retrieval_baseline.json`.

## Outputs
1. `best_model.pt`
2. `last_model.pt`
3. `training_history.json`
4. `metrics.json`
5. `val_metrics_eval.json`
6. `val_preview.png`

## Notes
1. This is a baseline for the optional task and does not use real multispectral channels.
2. It is intended as a retrieval pipeline starter for proposal/demo completeness.
