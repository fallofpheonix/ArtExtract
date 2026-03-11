# Full Zip Run Report

## Summary
1. Full end-to-end pipeline completed on zip-backed dataset.
2. Train samples: 11275; Val samples: 4707.
3. One unreadable image removed during manifest filtering.

## Metrics
1. Style top-1: 0.5587
2. Artist top-1: 0.4529
3. Artist top-5: 0.7999
4. Genre top-1: 0.5664
5. Style macro-F1: 0.2940
6. Artist macro-F1: 0.3896
7. Genre macro-F1: 0.4250

## Artifacts
1. `outputs/full_zip_run/best_model.pt`
2. `outputs/full_zip_run/training_history.json`
3. `outputs/full_zip_run/val_metrics.json`
4. `outputs/full_zip_run/val_predictions.csv`
5. `outputs/full_zip_run/style_outliers.csv`
6. `outputs/full_zip_run/training_curves_loss.png`
7. `outputs/full_zip_run/training_curves_metrics.png`
8. `outputs/full_zip_run/style_confusion_matrix.png`
9. `outputs/full_zip_run/artist_confusion_matrix_top30.png`
10. `outputs/full_zip_run/genre_confusion_matrix.png`

## Top Outlier Rows (style)
| index | class_id | score |
|---:|---:|---:|
| 3593 | 24 | 6.6069 |
| 2821 | 9 | 4.9038 |
| 4704 | 15 | 4.3311 |
| 1498 | 3 | 4.1934 |
| 4396 | 17 | 4.0530 |
| 2927 | 9 | 3.9784 |
| 1154 | 12 | 3.9573 |
| 1587 | 3 | 3.9538 |
| 1861 | 21 | 3.8815 |
| 331 | 12 | 3.7395 |
