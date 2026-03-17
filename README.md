# ArtExtract

Production-oriented research codebase for painting analysis workflows:
- Task 1: CRNN-based style/artist/genre classification
- Task 2: CLIP + FAISS similarity retrieval
- Multispectral track: property prediction and hidden-image detection

## Architecture (current)

The retrieval path is organized with explicit layers under `src/artextract/`:
- `core/`: domain-centric data shaping (manifest/metadata handling)
- `services/`: workflow orchestration for use-cases
- `api/`: CLI entrypoints
- `config/`: runtime configuration objects
- `utils/`: small shared primitives
- `tests/`: critical-path regression checks

Legacy modules under `retrieval/`, `models/`, `training/`, and `reconstruction/` remain active for compatibility.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional tooling:

```bash
pip install -e .[dev]
cp .env.example .env
```

## Run similarity retrieval

```bash
python3 scripts/run_similarity.py \
  --images-dir images \
  --opendata-dir nga_data \
  --out-dir analysis_out
```

## Tests

```bash
python3 -m unittest tests/test_similarity_service_layers.py
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Key decisions

- Keep legacy modules in place while moving new work to layered boundaries.
- Favor pragmatic defaults and fail-soft behavior (`unknown` labels, env-driven output path).
- Test only critical paths to avoid brittle, high-maintenance coverage on research code.
