# Troubleshooting

## 1. `torch/torchvision unavailable`
Cause:
1. Missing dependencies in active environment.

Fix:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. `missing image: data/raw/...`
Cause:
1. Manifest paths exist but image files were not downloaded.

Fix:
```bash
python scripts/download_images_from_manifest.py --manifest data/manifests/train_quick.csv --manifest data/manifests/val_quick.csv
```

Runner behavior:
1. Auto-picks `data/raw/wikiart` if present.
2. Use `IMAGES_ROOT=...` to force a root.

## 3. `broken data stream when reading image file`
Cause:
1. Corrupted/truncated image file in local archive.

Fix:
1. Keep `VALIDATE_IMAGES=1` (default) so unreadable rows are dropped pre-training.
2. Dataset loader falls back to nearby samples to prevent hard stop.

## 4. Very few rows in generated multitask manifest
Cause:
1. `style`, `artist`, `genre` split files do not fully overlap by `image_relpath`.

Check:
```bash
python scripts/build_manifest.py --split train --out data/manifests/train_multitask.csv
python scripts/dataset_stats.py --manifest data/manifests/train_multitask.csv
```

## 5. Slow/failed PDF notebook export
Cause:
1. Missing `playwright` browser runtime.

Fix:
```bash
pip install playwright
playwright install chromium
jupyter nbconvert --to webpdf notebooks/ArtExtract_Task1_CRNN.executed.ipynb --output ArtExtract_Task1_CRNN.executed.pdf
```

## 6. Poor metrics in quick run
Cause:
1. Small subset size.
2. Few epochs.
3. CPU-only training.

Fix:
1. Increase `EPOCHS`.
2. Increase `MAX_IMAGES`.
3. Run on GPU with `--device cuda`.
4. Train with larger/cleaner dataset coverage.

## 7. Shape mismatch when editing config
Cause:
1. Incompatible `global_dim` with backbone output.

Fix:
1. Model now projects backbone output to `global_dim`; mismatch is handled automatically.
2. Keep `cnn_backbone` in `{resnet18,resnet34,resnet50}`.
