# Task 1 Scaled Run Report

## Run Context
- Mode: full-manifest selection with max image cap (scaled run)
- Train samples: 1031
- Val samples: 357
- Epochs: 1
- Checkpoint: `outputs/quick_cpu_scaled/best_model.pt`

## Validation Metrics
- Style Top-1: 0.4706
- Style Weighted-F1: 0.3692
- Artist Top-1: 0.3277
- Artist Top-5: 0.7507
- Genre Top-1: 0.2689

## Outlier Summary
- Style outliers: 37
- idx=36 score=3.377 image=`Art_Nouveau_Modern/raphael-kirchner_boys-and-girls-at-sea-3.jpg`
- idx=139 score=3.324 image=`Art_Nouveau_Modern/raphael-kirchner_girls-with-good-luck-charms-7.jpg`
- idx=304 score=3.217 image=`Analytical_Cubism/pablo-picasso_female-nude.jpg`
- idx=155 score=3.089 image=`Art_Nouveau_Modern/raphael-kirchner_boys-and-girls-at-sea-8.jpg`
- idx=50 score=2.846 image=`Art_Nouveau_Modern/raphael-kirchner_greek-virgins-1900-8.jpg`
- idx=81 score=2.778 image=`Art_Nouveau_Modern/raphael-kirchner_boys-and-girls-at-sea-5.jpg`
- idx=114 score=2.738 image=`Art_Nouveau_Modern/raphael-kirchner_maid-of-athens-1900-5.jpg`
- idx=221 score=2.671 image=`Baroque/rembrandt_batavernas-trohetsed-1662.jpg`
- idx=294 score=2.637 image=`Symbolism/nicholas-roerich_the-pact-of-culture-study-1931.jpg`
- idx=253 score=2.514 image=`Baroque/rembrandt_a-woman-sitting-up-in-bed.jpg`