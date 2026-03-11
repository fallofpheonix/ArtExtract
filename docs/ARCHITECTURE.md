# Architecture

## Unified Multispectral Stack

Input contract per sample:
1. `x`: tensor `[C,H,W]` where `C=len(channels)`
2. `channel_mask`: tensor `[C]` (1=present, 0=missing)
3. `targets` by enabled tasks:
   - properties: `pigments`, `damage`, `restoration`
   - hidden: `hidden_image`
   - reconstruction: `hidden_gt`

## Model
`src/artextract/models/multispectral.py`

1. Shared encoder consumes masked input `x * channel_mask`.
2. Pooled embedding concatenates channel mask for modality-awareness.
3. Optional heads:
   - property head: pigments + damage + restoration logits
   - hidden head: binary hidden-image logit
   - reconstruction head: U-Net decoder (`src/artextract/reconstruction/unet.py`)

## Loss
Total loss:
`L = λ_property * L_property + λ_hidden * L_hidden + λ_reconstruction * L_reconstruction`

Where:
1. `L_property`: average of BCE losses for pigments/damage/restoration
2. `L_hidden`: BCEWithLogits
3. `L_reconstruction`: `L1 + MSE`

Weights are configured in `configs/multispectral_baseline.json`.

## Evaluation Artifacts
`src/artextract/evaluation/multispectral.py` writes:
1. `metrics.json`
2. `run_meta.json`
3. `confusion_matrix.png` (hidden task)
4. `roc_curve.png` (hidden task)
5. `recon_examples/*` (reconstruction task)

## Legacy Tracks
1. CRNN RGB classification (`train_crnn.py`) remains unchanged.
2. Hidden retrieval baseline (`train_hidden_retrieval.py`) remains unchanged.
3. Similarity CLIP/FAISS moved under `src/artextract/similarity` and executed via `scripts/run_similarity.py`.
