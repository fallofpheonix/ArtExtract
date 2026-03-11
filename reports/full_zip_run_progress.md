# Full Zip Run Progress (2026-03-11)

## Execution Status
1. End-to-end pipeline run: completed.
2. Data source: local `wikiart.zip` extracted to `data/raw/wikiart`.
3. Train manifest rows: 11,275 (1 unreadable row removed).
4. Val manifest rows: 4,707.
5. Outputs directory: `outputs/full_zip_run`.

## Validation Metrics
1. Style top-1: 0.5587422987040578
2. Artist top-1: 0.4529424261737837
3. Artist top-5: 0.7998725302740599
4. Genre top-1: 0.5663904822604632
5. Outliers exported: 477 (`outputs/full_zip_run/style_outliers.csv`).

## Completion Score (Out of 100)
1. Project structure and reproducible scripts: 100/100.
2. Data pipeline integration (zip + filtering + manifests): 95/100.
3. Training/evaluation pipeline execution: 100/100.
4. Documentation quality and runbook coverage: 92/100.
5. Research target quality (accuracy vs expected strong baseline): 72/100.

Overall weighted completion: 92/100.

## Remaining Gap To 100
1. Add augmentation + class balancing + uncertainty weighting.
2. Run longer training schedule (>=10 epochs) on full intersection set.
3. Execute notebook with generated plots/tables from `outputs/full_zip_run`.
4. Add confusion matrix figures and outlier examples to submission report.
