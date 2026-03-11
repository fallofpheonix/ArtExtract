# Architecture

## Model
`src/artextract/models/crnn.py` defines `CRNNMultiTask`.

Input:
1. Tensor shape `[B, 3, H, W]`.

Output dictionary:
1. `style` logits `[B, N_style]`
2. `artist` logits `[B, N_artist]`
3. `genre` logits `[B, N_genre]`
4. `embedding` fused feature `[B, fusion_dim]`

## Data Flow
1. Global branch:
   - `ResNet18` backbone with `fc=Identity`.
   - Produces global feature vector (default 512-d).
2. Patch branch:
   - Uniform `patch_grid x patch_grid` split from input image.
   - Shared lightweight CNN encoder per patch.
   - Linear projection to `patch_dim`.
   - BiGRU over patch sequence.
3. Fusion:
   - Concatenate global feature and GRU feature.
   - Dropout.
   - Three linear heads for style/artist/genre.

## Default Config Parameters
From `configs/baseline.json` / `configs/quick_cpu.json`:
1. `patch_grid`
2. `global_dim`
3. `patch_dim`
4. `rnn_hidden`
5. `dropout`

## Training Objective
`L_total = L_style + L_artist + L_genre`

Each task loss:
1. CrossEntropyLoss.

## Complexity (per batch)
1. Patch extraction path: `O(B * P * C_patch)` where `P = patch_grid^2` and `C_patch` is patch encoder cost.
2. GRU path: `O(B * P * H * (E + H))`, with `E=patch_dim`, `H=rnn_hidden`.
3. Memory grows linearly with batch size and number of patches.

## Current Engineering Constraints
1. `global_dim` must match backbone output dimension (ResNet18 default 512) for head shape consistency.
2. Patch processing is vectorized (no Python loop over patches), but still scales linearly with patch count.
