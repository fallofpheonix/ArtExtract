# CLI Reference

## Unified Multispectral CLIs

## `scripts/train.py`
```bash
python scripts/train.py \
  [--manifest data/manifests/multispectral.csv] \
  [--channels rgb,ir,uv,xray] \
  [--tasks properties,hidden,reconstruction] \
  [--config configs/multispectral_baseline.json] \
  [--out-dir reports/run_id] \
  [--device cpu|cuda|mps]
```

Synthetic fallback mode (when real multispectral manifest is unavailable):
```bash
python scripts/train.py \
  --synthetic-images-root /path/to/rgb_images \
  --synthetic-out-root data/synthetic_multispectral \
  --synthetic-max-samples 200
```

## `scripts/eval.py`
```bash
python scripts/eval.py \
  --manifest reports/run_id/resolved_manifest.csv \
  --checkpoint reports/run_id/best_model.pt \
  --pigments-vocab reports/run_id/pigments_vocab.json \
  --channels rgb,ir,uv,xray \
  --tasks properties,hidden,reconstruction \
  --config configs/multispectral_baseline.json \
  --out-dir reports/run_id \
  [--device cpu|cuda|mps]
```

## `scripts/generate_synthetic_multispectral.py`
```bash
python scripts/generate_synthetic_multispectral.py \
  --images-root /path/to/images \
  [--out-root data/synthetic_multispectral] \
  [--channels rgb,ir,uv,xray] \
  [--image-size 128] [--max-samples 200] [--train-ratio 0.8] [--seed 42]
```

## `scripts/run_reconstruction.py`
```bash
python scripts/run_reconstruction.py \
  --images-root /path/to/images \
  [--channels rgb,ir,uv,xray] \
  [--config configs/multispectral_baseline.json] \
  [--out-dir reports/reconstruction_run] \
  [--max-samples 300]
```

## `scripts/run_similarity.py`
```bash
python scripts/run_similarity.py --images-dir images --opendata-dir nga_data
```

## Legacy Baselines (Preserved)

## `scripts/run_quick_pipeline.sh`
```bash
bash scripts/run_quick_pipeline.sh
```

Key env vars:
1. `QUICK_MODE` (`1` sampled quick, `0` full-manifest)
2. `PER_STYLE_TRAIN`, `PER_STYLE_VAL` (quick mode caps)
3. `EPOCHS`, `BATCH_SIZE`, `MAX_IMAGES`
4. `IMAGES_ROOT`, `SKIP_DOWNLOAD`, `VALIDATE_IMAGES`

## `scripts/run_hidden_retrieval_pipeline.sh`
```bash
bash scripts/run_hidden_retrieval_pipeline.sh
```
