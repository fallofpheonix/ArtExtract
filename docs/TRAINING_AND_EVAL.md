# Training and Evaluation

## Unified Multispectral Flow

1. Prepare manifest (CSV/JSONL) with fields:
   - `sample_id`, `split`, `channels`, `width`, `height`
   - channel path columns (`rgb_path`, `ir_path`, `uv_path`, `xray_path`, ...)
   - labels: `pigments`, `damage`, `restoration`, `hidden_image`
   - optional `hidden_gt_path`
2. Train with `scripts/train.py`.
3. Evaluate with `scripts/eval.py`.

## Train Example
```bash
python scripts/train.py \
  --manifest data/manifests/multispectral.csv \
  --channels rgb,ir,uv,xray \
  --tasks properties,hidden,reconstruction \
  --config configs/multispectral_baseline.json \
  --out-dir reports/run_ms
```

## Eval Example
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

## No Real Multispectral Data
Use synthetic harness:
```bash
python scripts/generate_synthetic_multispectral.py --images-root /path/to/rgb/images
```
or directly via `scripts/train.py --synthetic-images-root ...`.

## Determinism
Deterministic controls are enabled in training:
1. python/numpy/torch seed set from config
2. `torch.backends.cudnn.deterministic=True`
3. `torch.backends.cudnn.benchmark=False`

## Metrics
1. Properties: pigments macro-F1, damage/restoration accuracy/precision/recall/F1
2. Hidden detection: accuracy, precision, recall, F1, AUC
3. Reconstruction: PSNR, SSIM

Absent-task metrics are omitted from `metrics.json`.
