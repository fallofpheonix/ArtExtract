# ArtExtract Task 2: Painting Similarity

HumanAI/CERN evaluation project for painting similarity retrieval using National Gallery of Art metadata + images.

## What This Repository Contains
- `ArtExtract_Task2_Similarity.ipynb`: baseline (handcrafted features).
- `ArtExtract_Task2_CLIP_FAISS.ipynb`: notebook variant for CLIP+FAISS.
- `pipeline_clip_faiss.py`: production-style CLI pipeline.
- `download_images.py`: NGA image downloader from metadata URLs.
- `requirements.txt`, `requirements_clip_faiss.txt`: dependency sets.
- `report.pdf`: submission report.
- `RUN.md`: command-only execution sequence.
- `TROUBLESHOOTING.md`: operational fixes.

## Dataset Model
Source: [National Gallery of Art Open Data](https://github.com/NationalGalleryOfArt/opendata)

Important constraints:
- NGA repo provides metadata + URLs, not image binaries.
- Minimum required metadata files:
  - `nga_data/data/objects.csv`
  - `nga_data/data/published_images.csv`

`download_images.py` can auto-fetch `published_images.csv` if `nga_data` is missing.

## Quick Start
### 1) Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -r requirements_clip_faiss.txt
```

### 2) Download images
```bash
python download_images.py \
  --opendata-dir nga_data \
  --output-dir images \
  --max-images 5000 \
  --workers 24
```

### 3) Run advanced retrieval pipeline
```bash
python pipeline_clip_faiss.py \
  --images-dir images \
  --opendata-dir nga_data \
  --max-images 1200 \
  --label-col classification \
  --top-k 10 \
  --clusters 20 \
  --tsne-sample 800 \
  --out-dir analysis_out
```

## Output Artifacts
`analysis_out/` includes:
- `embedding_metadata.csv`
- `cluster_counts.csv`
- `cluster_representatives.csv`
- `cluster_label_composition.csv`
- `pca_clusters.png`

## Metrics Reported
- Precision@K
- Recall@K
- mAP@K
- nDCG@K

## Baseline Notebook Path (Optional)
```bash
python -m jupyter nbconvert --execute --to notebook --inplace ArtExtract_Task2_Similarity.ipynb
python -m jupyter nbconvert --to html ArtExtract_Task2_Similarity.ipynb
```

PDF export:
```bash
pip install "nbconvert[webpdf]"
python -m playwright install chromium
python -m jupyter nbconvert --to webpdf ArtExtract_Task2_Similarity.ipynb
```

## Runtime Defaults and Stability
- Default device: CPU.
- CUDA auto-detected if available.
- MPS is opt-in only: `ENABLE_MPS=1`.
- Default CLIP backend: `open_clip` (`ViT-B-32`, `laion2b_s34b_b79k`).
- HF cache directory: `.hf_cache/hub`.

## Submission Checklist
- `README.md`
- `requirements.txt`
- `requirements_clip_faiss.txt`
- `download_images.py`
- `ArtExtract_Task2_Similarity.ipynb`
- `report.pdf`

## Verification
```bash
python -m py_compile download_images.py pipeline_clip_faiss.py
python -m jupyter --version
```

For known failures and fixes: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
