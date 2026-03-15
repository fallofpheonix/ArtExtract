# Task 1

Task 1 implements the classification and outlier-analysis track for ArtExtract.
The current target is a convolutional-recurrent pipeline over WikiArt-style RGB inputs with multi-head outputs for style, artist, and genre.

## Dataset

- Name: ArtGAN WikiArt Dataset
- Link: <https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md>

Local metadata contract:

1. `data/metadata/wikiart_csv/style_{train,val}.csv`
2. `data/metadata/wikiart_csv/artist_{train,val}.csv`
3. `data/metadata/wikiart_csv/genre_{train,val}.csv`

## Scope

1. Multi-head classification for style, artist, and genre
2. Outlier discovery for samples that do not fit the assigned class distribution
3. Evaluation under long-tail label imbalance
4. Curator-review artifacts for failure analysis

## Relevant Code

1. `scripts/train_crnn.py`
2. `scripts/evaluate_model.py`
3. `scripts/detect_outliers.py`
4. `src/artextract/models/crnn.py`
5. `src/artextract/data/multitask.py`
6. `src/artextract/training/outliers.py`
7. `configs/baseline.json`
8. `configs/baseline.yaml`

## Typical Workflow

```bash
python scripts/train_crnn.py --config configs/baseline.json
python scripts/evaluate_model.py --config configs/baseline.json
python scripts/detect_outliers.py --config configs/baseline.json
```

## Expected Outputs

1. Task checkpoints and training history under the selected output directory
2. Validation metrics for style, artist, and genre
3. Ranked outlier candidates for review

## Evaluation Focus

1. style top-1
2. artist top-1 and top-5
3. genre top-1
4. macro-F1 and class-balance diagnostics where added by future work

## Next Development Targets

1. stronger outlier ranking explanations
2. better calibration under long-tail artist labels
3. unified artifact schema with the other repository tracks
