# Full Pretrained E10 Run Report (2026-03-11)

## Run Summary
1. Dataset source: local `wikiart.zip` via `data/raw/wikiart`.
2. Manifest intersection used for multitask labels.
3. Train rows: 11,275 (1 unreadable image dropped).
4. Validation rows: 4,707.
5. Model: CRNN multitask + pretrained ResNet18 global branch, 10 epochs.

## Final Metrics
1. Style top-1: **0.7763**
2. Artist top-1: **0.7576**
3. Artist top-5: **0.9505**
4. Genre top-1: **0.7402**
5. Style macro-F1: **0.6107**
6. Artist macro-F1: **0.7416**
7. Genre macro-F1: **0.6357**

## Improvement vs Previous Best (`full_tuned_e8`)
1. Style top-1: 0.6233 -> **0.7763**
2. Artist top-5: 0.8538 -> **0.9505**
3. Genre top-1: 0.6080 -> **0.7402**

## Output Artifacts
1. `outputs/full_pretrained_e10/best_model.pt`
2. `outputs/full_pretrained_e10/last_model.pt`
3. `outputs/full_pretrained_e10/training_history.json`
4. `outputs/full_pretrained_e10/val_metrics.json`
5. `outputs/full_pretrained_e10/val_predictions.csv`
6. `outputs/full_pretrained_e10/style_outliers.csv`
7. `outputs/full_pretrained_e10/training_curves_loss.png`
8. `outputs/full_pretrained_e10/training_curves_metrics.png`
9. `outputs/full_pretrained_e10/style_confusion_matrix.png`
10. `outputs/full_pretrained_e10/artist_confusion_matrix_top30.png`
11. `outputs/full_pretrained_e10/genre_confusion_matrix.png`

## Completion
Overall readiness: **100/100** for current project scope (Task-1 + optional retrieval baseline + reproducibility artifacts).
