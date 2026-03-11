# Getting Started

## 1) Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Quick Unified Smoke Run (Synthetic Multispectral)
```bash
python scripts/train.py \
  --synthetic-images-root /path/to/rgb/images \
  --channels rgb,ir,uv,xray \
  --tasks properties,hidden,reconstruction \
  --config configs/multispectral_baseline.json \
  --out-dir reports/smoke

python scripts/eval.py \
  --manifest reports/smoke/resolved_manifest.csv \
  --checkpoint reports/smoke/best_model.pt \
  --pigments-vocab reports/smoke/pigments_vocab.json \
  --channels rgb,ir,uv,xray \
  --tasks properties,hidden,reconstruction \
  --config configs/multispectral_baseline.json \
  --out-dir reports/smoke
```

## 3) Task2 Similarity Run
```bash
pip install -r requirements_similarity.txt
python scripts/run_similarity.py --images-dir images --opendata-dir nga_data
```

## 4) Legacy Baselines (Optional)
```bash
bash scripts/run_quick_pipeline.sh
bash scripts/run_hidden_retrieval_pipeline.sh
```
