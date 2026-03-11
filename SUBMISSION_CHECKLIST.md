# Submission Checklist (HumanAI: Painting in a Painting)

## Email
1. Send to `human-ai@cern.ch`
2. Subject: `Evaluation Test: ArtExtract`

## Required Package
1. CV (PDF)
2. GitHub repository link (this repo)
3. Jupyter notebook (`.ipynb`)
4. PDF export of notebook with outputs

## Recommended Technical Attachments
1. `reports/<run_id>/metrics.json`
2. `reports/<run_id>/run_meta.json`
3. `reports/<run_id>/confusion_matrix.png` (if hidden detection run)
4. `reports/<run_id>/roc_curve.png` (if hidden detection run)
5. `reports/<run_id>/recon_examples/*` (if reconstruction run)

## Final Validation Before Sending
1. Run unit tests
2. Run multispectral train/eval smoke pipeline
3. Verify README commands match actual CLI behavior
4. Confirm no large raw data/caches are tracked in git
