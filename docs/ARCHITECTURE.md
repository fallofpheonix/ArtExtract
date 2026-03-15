# Architecture

ArtExtract is organized as a multi-track repository with shared data utilities and separate execution paths.

## Top-Level Modules

1. `src/artextract/data`
   Dataset loaders, manifest parsing, WikiArt contracts, and synthetic multispectral generation.
2. `src/artextract/models`
   CRNN classification model and the unified multispectral multitask model.
3. `src/artextract/training`
   Training loops, metrics aggregation, and outlier helpers.
4. `src/artextract/evaluation`
   Multispectral evaluation pipeline and report artifact generation.
5. `src/artextract/reconstruction`
   U-Net style decoder used for optional hidden-image reconstruction.
6. `src/artextract/retrieval`
   Hidden-retrieval baseline modules.
7. `src/artextract/similarity`
   CLIP plus FAISS similarity pipeline and NGA helper utilities.
8. `scripts/`
   Stable CLI entrypoints. These are the public execution boundary for the repository.

## Track Architecture

### Task 1: Classification and Outliers

Data flow:

1. WikiArt split CSVs in `data/metadata/wikiart_csv/` are joined into train and validation manifests.
2. `scripts/train_crnn.py` loads `configs/baseline.json` or `configs/baseline.yaml`.
3. `src/artextract/data/multitask.py` builds RGB image batches with labels for style, artist, and genre.
4. `src/artextract/models/crnn.py` emits one head per target.
5. `scripts/evaluate_model.py` computes metrics.
6. `scripts/detect_outliers.py` ranks samples whose embeddings or logits deviate from expected class structure.

Invariant:

1. Each sample must resolve to a valid image path and integer class IDs for all enabled targets.

### Task 2: Similarity Retrieval

Data flow:

1. NGA metadata is read from a local clone of the opendata repository.
2. Image paths are matched against the local image directory.
3. `scripts/run_similarity.py` invokes `src/artextract/similarity/clip_faiss.py`.
4. The pipeline computes CLIP embeddings, builds a FAISS index, runs nearest-neighbor search, and emits ranking artifacts.

Invariant:

1. Metadata and local image filenames must be joinable through a stable identifier; retrieval quality is undefined otherwise.

### Unified Multispectral Stack

Input contract per sample:

1. `x`: tensor `[C,H,W]`, where `C` equals the selected channel count
2. `channel_mask`: tensor `[C]`, where `1` means present and `0` means missing
3. Optional targets:
   - `pigments`
   - `damage`
   - `restoration`
   - `hidden_image`
   - `hidden_gt`

Model path:

1. `src/artextract/models/multispectral.py`
2. Shared encoder consumes masked input `x * channel_mask`.
3. Pooled embedding is concatenated with the channel mask to keep modality awareness explicit.
4. Optional heads:
   - property head
   - hidden-image detection head
   - reconstruction decoder in `src/artextract/reconstruction/unet.py`

Loss:

`L = lambda_property * L_property + lambda_hidden * L_hidden + lambda_reconstruction * L_reconstruction`

Where:

1. `L_property` is the aggregate property loss
2. `L_hidden` is binary classification loss
3. `L_reconstruction` is `L1 + MSE`

Weights are configured in `configs/multispectral_baseline.json`.

## Output Contracts

The unified multispectral evaluation path writes:

1. `metrics.json`
2. `run_meta.json`
3. `confusion_matrix.png` when hidden detection is enabled
4. `roc_curve.png` when hidden detection is enabled
5. `recon_examples/` when reconstruction is enabled

These are generated artifacts, not primary documentation.
