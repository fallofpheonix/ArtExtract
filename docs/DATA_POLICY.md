# Data Policy

This repository mixes public metadata, local image paths, and synthetic multispectral data. Data handling must stay explicit.

## Dataset Sources

### Task 1

1. Dataset: ArtGAN WikiArt
2. Reference: <https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md>
3. Local metadata lives under `data/metadata/wikiart_csv/`

### Task 2

1. Dataset: National Gallery of Art Open Data
2. Reference: <https://github.com/NationalGalleryOfArt/opendata>
3. The repository does not vendor NGA images; metadata and local downloads are joined at runtime

### Multispectral Track

1. No public multispectral dataset is bundled here
2. Training expects a manifest that resolves channel paths on local storage
3. Synthetic multispectral generation is provided for smoke testing and model contract validation

## Repository Data Layout

1. `data/manifests/`
   CSV manifests used by scripts and tests
2. `data/metadata/wikiart_csv/`
   split definitions and class maps for Task 1
3. `data/templates/provenance_template.csv`
   provenance and redistribution template

## Manifest Contracts

### Task 1 Manifests

Required fields:

1. `image_relpath`
2. `style_label`
3. `artist_label`
4. `genre_label`

### Multispectral Manifests

Expected fields depend on enabled tasks, but the canonical schema includes:

1. `sample_id`
2. `split`
3. channel path columns such as `rgb_path`, `ir_path`, `uv_path`, `xray_path`
4. image geometry fields
5. task labels such as `pigments`, `damage`, `restoration`, `hidden_image`
6. optional `hidden_gt_path`

## Provenance and Redistribution

The hard policy is defined in [../DATA_RIGHTS.md](../DATA_RIGHTS.md).

Operational rules:

1. keep provenance for every distributed sample
2. tag redistribution status explicitly
3. filter release bundles before publication
4. keep takedown paths possible through traceable sample metadata

Release gate:

```bash
python scripts/filter_redistributable.py \
  --input data/templates/provenance_template.csv \
  --out-allowed outputs/provenance_allowed.csv \
  --out-blocked outputs/provenance_blocked.csv
```

## Non-Goals

1. This repository is not a raw image mirror.
2. Copyright status is not inferred from source availability.
3. Submission artifacts do not imply redistribution rights for training data.
