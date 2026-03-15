# Task 2

Task 2 implements embedding-based painting similarity retrieval.
The current baseline uses CLIP embeddings and FAISS indexing over locally downloaded paintings with NGA metadata.

## Dataset

- Name: National Gallery of Art Open Data
- Link: <https://github.com/NationalGalleryOfArt/opendata>

Runtime assumption:

1. the NGA metadata repository is available locally at `nga_data/`
2. referenced images have been downloaded into a local image directory

## Scope

1. embedding extraction
2. FAISS nearest-neighbor retrieval
3. top-k retrieval analysis
4. optional clustering and qualitative exploration
5. future metadata-aware ranking and filtering

## Relevant Code

1. `scripts/run_similarity.py`
2. `src/artextract/similarity/clip_faiss.py`
3. `src/artextract/similarity/download_images.py`
4. `requirements_similarity.txt`
5. `test_results/task2/`

## Typical Workflow

```bash
pip install -r requirements_similarity.txt
python scripts/run_similarity.py --images-dir images --opendata-dir nga_data
```

## Expected Outputs

1. embedding metadata export
2. retrieval rankings
3. optional clustering artifacts
4. runtime logs and evaluation summaries

## Current Boundary

1. the first run may fail during remote CLIP weight download if the Hugging Face transfer times out
2. this is an environment/runtime issue, not a repository structure issue

## Next Development Targets

1. cached model bootstrap for deterministic first-run behavior
2. stronger retrieval metrics and benchmark sets
3. metadata-aware filtering and reranking
