# Task 1 Final Perfected Run Report

## Run Context
- Mode: full-manifest mode (`QUICK_MODE=0`) with cached dataset
- Train samples: 1031
- Val samples: 357
- Epochs: 5
- Checkpoint: `outputs/final_perfected/best_model.pt`

## Validation Metrics
- Style Top-1: 0.5686
- Style Weighted-F1: 0.4699
- Artist Top-1: 0.5238
- Artist Top-5: 0.8431
- Artist Weighted-F1: 0.4630
- Genre Top-1: 0.4146
- Genre Weighted-F1: 0.3818

## Outlier Summary
- Style outliers detected: 37
- idx=79 score=4.570 image=`Art_Nouveau_Modern/nicholas-roerich_untitled-1908.jpg`
- idx=108 score=4.365 image=`Art_Nouveau_Modern/boris-kustodiev_bather-seated-on-the-shore-1926.jpg`
- idx=266 score=2.705 image=`Expressionism/martiros-saryan_illustration-to-yeghishe-charents-country-of-nairi-escape-of-nairyans-1933.jpg`
- idx=338 score=2.393 image=`Northern_Renaissance/albrecht-durer_portrait-of-a-man-with-baret-and-scroll-1521.jpg`
- idx=303 score=2.348 image=`Analytical_Cubism/pablo-picasso_still-life-with-bottle-of-anis-del-mono-1909.jpg`
- idx=169 score=2.280 image=`Art_Nouveau_Modern/nicholas-roerich_vignette-for-book-n-k-roerich-1918-1.jpg`
- idx=348 score=2.279 image=`Naive_Art_Primitivism/boris-kustodiev_easter-procession-1915-1.jpg`
- idx=257 score=2.209 image=`Baroque/rembrandt_drawing-of-the-last-supper-1635.jpg`
- idx=66 score=2.208 image=`Art_Nouveau_Modern/nicholas-roerich_the-painting-of-of-st-anastasia-1913.jpg`
- idx=244 score=2.144 image=`Baroque/rembrandt_abraham-s-sacrifice-1655.jpg`
- idx=238 score=2.120 image=`Baroque/rembrandt_christ-driving-the-moneychangers-from-the-temple-1635.jpg`
- idx=50 score=2.071 image=`Art_Nouveau_Modern/raphael-kirchner_greek-virgins-1900-8.jpg`