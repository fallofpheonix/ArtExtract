# Development

This document defines the supported developer workflow for the current repository.

## Environment

Base environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -r requirements.lock.txt
pip install -r requirements_similarity.txt
```

## Entry Points

### Task 1

```bash
python scripts/train_crnn.py --config configs/baseline.json
python scripts/evaluate_model.py --config configs/baseline.json
python scripts/detect_outliers.py --config configs/baseline.json
```

### Task 2

```bash
python scripts/run_similarity.py --images-dir images --opendata-dir nga_data
```

### Multispectral Training

```bash
python scripts/train.py \
  --manifest data/manifests/multispectral.csv \
  --channels rgb,ir,uv,xray \
  --tasks properties,hidden,reconstruction \
  --config configs/multispectral_baseline.json \
  --out-dir reports/run_ms
```

### Multispectral Evaluation

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

### Synthetic Multispectral Data

```bash
python scripts/generate_synthetic_multispectral.py \
  --images-root /path/to/images \
  --out-root data/synthetic_multispectral \
  --channels rgb,ir,uv,xray \
  --max-samples 500
```

### Reconstruction Smoke Path

```bash
python scripts/run_reconstruction.py --images-root /path/to/images
```

## Tests

Automated regression:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Coverage intent:

1. manifest and loader contracts
2. multispectral model forward and gradient sanity
3. CLI smoke path for train and eval

## Generated Artifacts

Generated outputs belong in `reports/` or `test_results/`, not inside source documentation.

Expected multispectral outputs:

1. `metrics.json`
2. `run_meta.json`
3. `confusion_matrix.png`
4. `roc_curve.png`
5. `recon_examples/`

## Development Rules

1. Update the primary Markdown files before adding new ad hoc notes.
2. Keep source notebooks only; do not commit rendered HTML, PDF, or executed notebook copies unless explicitly required for submission.
3. Keep generated reports reproducible from scripts instead of storing many static snapshots.
4. Do not commit raw image datasets or non-redistributable assets.

## Known Operational Boundaries

1. Task 2 may fail during first-time CLIP model download if the Hugging Face fetch times out.
2. Real multispectral data is not bundled in the repository; synthetic generation is the default fallback path.
3. YAML configs are supported, but JSON configs are the default interface across scripts.
