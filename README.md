# ArtExtract

ArtExtract is the working repository for the HumanAI "Painting in a Painting" project.
The codebase currently supports four development tracks:

1. Task 1: CRNN-based painting attribute classification and outlier analysis
2. Task 2: embedding-based painting similarity retrieval
3. Multispectral property prediction and hidden-image detection
4. Optional hidden-image reconstruction baseline

## Primary Documentation

These files are the authoritative project references and should be updated first when behavior changes:

1. [README.md](README.md)
2. [docs/README.md](docs/README.md)
3. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
4. [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
5. [docs/DATA_POLICY.md](docs/DATA_POLICY.md)
6. [docs/ROADMAP.md](docs/ROADMAP.md)
7. [task1/README.md](task1/README.md)
8. [task2/README.md](task2/README.md)

Redundant generated reports, exports, and stale documentation are intentionally removed from the primary information path.

## Repository Layout

```text
ArtExtract/
  configs/           configuration files for training and retrieval pipelines
  data/              manifests, metadata splits, and provenance templates
  docs/              authoritative design, data, and development documents
  notebooks/         source notebooks only
  reports/           generated run artifacts; not canonical documentation
  scripts/           runnable entrypoints
  src/artextract/    library code
  task1/             task-specific documentation for classification/outliers
  task2/             task-specific documentation for similarity retrieval
  test_results/      submission evidence and sample outputs
  tests/             automated regression coverage
```

## Datasets

1. Task 1: ArtGAN WikiArt
   Link: <https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md>
2. Task 2: National Gallery of Art Open Data
   Link: <https://github.com/NationalGalleryOfArt/opendata>
3. Multispectral track: local or partner-provided manifests only; no redistributable dataset is bundled here

Rights and provenance rules are defined in [DATA_RIGHTS.md](DATA_RIGHTS.md) and [docs/DATA_POLICY.md](docs/DATA_POLICY.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional:

```bash
pip install -r requirements.lock.txt
pip install -r requirements_similarity.txt
```

## Canonical Workflows

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

### Multispectral Train and Eval

```bash
python scripts/train.py \
  --manifest data/manifests/multispectral.csv \
  --channels rgb,ir,uv,xray \
  --tasks properties,hidden,reconstruction \
  --config configs/multispectral_baseline.json \
  --out-dir reports/run_ms

python scripts/eval.py \
  --manifest reports/run_ms/resolved_manifest.csv \
  --checkpoint reports/run_ms/best_model.pt \
  --pigments-vocab reports/run_ms/pigments_vocab.json \
  --channels rgb,ir,uv,xray \
  --tasks properties,hidden,reconstruction \
  --config configs/multispectral_baseline.json \
  --out-dir reports/run_ms
```

### Synthetic Multispectral Smoke Run

```bash
python scripts/run_reconstruction.py --images-root /path/to/images
```

## Testing

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Current automated coverage targets manifest loading, model forward contracts, and the unified multispectral CLI smoke path.

## Submission Evidence

Evaluation artifacts are retained under `test_results/` because they are part of the HumanAI submission package. They are not treated as primary documentation.
