# Task 1 Quick Run Report

## Run Context
- Config: `configs/quick_cpu.json`
- Train samples: 396
- Val samples: 130
- Epochs: 3 (CPU)
- Checkpoint: `outputs/quick_cpu/best_model.pt`

## Validation Metrics
- Style Top-1: 0.2769
- Style Macro-F1: 0.2395
- Artist Top-1: 0.2923
- Artist Top-5: 0.6769
- Artist Weighted-F1: 0.2339
- Genre Top-1: 0.3923
- Genre Macro-F1: 0.2645

## Top Style Outliers (by embedding distance z-score)
- idx=49 score=2.964 image=`Expressionism/martiros-saryan_illustration-to-yeghishe-charents-country-of-nairi-escape-of-nairyans-1933.jpg` true=Expressionism pred=Cubism
- idx=36 score=2.685 image=`Cubism/pablo-picasso_pitcher-and-bowls-1908.jpg` true=Cubism pred=Baroque
- idx=17 score=2.358 image=`Art_Nouveau_Modern/nicholas-roerich_vignette-for-book-n-k-roerich-1918-18.jpg` true=Art_Nouveau pred=Cubism
- idx=92 score=2.293 image=`Northern_Renaissance/albrecht-durer_portrait-of-a-man-with-baret-and-scroll-1521.jpg` true=Northern_Renaissance pred=Baroque
- idx=83 score=2.011 image=`Naive_Art_Primitivism/pablo-picasso_a-simple-meal-1904-1.jpg` true=Naive_Art_Primitivism pred=Art_Nouveau
- idx=94 score=1.982 image=`Pointillism/camille-pissarro_haymakers-resting-1891.jpg` true=Pointillism pred=Pointillism
- idx=107 score=1.967 image=`Post_Impressionism/pyotr-konchalovsky_france-mountain-lavender-1908.jpg` true=Post_Impressionism pred=Post_Impressionism
- idx=118 score=1.881 image=`Romanticism/gustave-dore_don-quixote-99.jpg` true=Romanticism pred=Naive_Art_Primitivism
- idx=112 score=1.848 image=`Realism/pyotr-konchalovsky_children-in-the-park-1940.jpg` true=Realism pred=Analytical_Cubism
- idx=63 score=1.639 image=`Fauvism/pyotr-konchalovsky_bullfighting-amateur-1910.jpg` true=Fauvism pred=Cubism

## Artifacts
- `outputs/quick_cpu/training_history.json`
- `outputs/quick_cpu/val_metrics.json`
- `outputs/quick_cpu/val_predictions.csv`
- `outputs/quick_cpu/val_embeddings.npy`
- `outputs/quick_cpu/val_style_labels.npy`
- `outputs/quick_cpu/style_outliers.csv`
- `outputs/quick_cpu/style_outliers_annotated.csv`