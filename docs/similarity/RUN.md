# RUN

## 1) Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -r requirements_clip_faiss.txt
```

## 2) Download Dataset Images
```bash
python download_images.py --opendata-dir nga_data --output-dir images --max-images 5000 --workers 24
```

If `nga_data` is missing, the downloader auto-fetches `published_images.csv`.

## 3) Run Baseline Notebook
```bash
python -m jupyter nbconvert --execute --to notebook --inplace ArtExtract_Task2_Similarity.ipynb
python -m jupyter nbconvert --to html ArtExtract_Task2_Similarity.ipynb
```

## 4) Export Baseline PDF
```bash
pip install "nbconvert[webpdf]"
python -m playwright install chromium
python -m jupyter nbconvert --to webpdf ArtExtract_Task2_Similarity.ipynb
```

## 5) Run Advanced CLIP+FAISS Pipeline
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
