---
layout: page
title: "Painting in a Painting (ArtExtract @ HumanAI)"
---

## Painting in a Painting: Hidden Image Discovery with AI

### Description
Recent multispectral and X-ray imaging methods allow experts to discover hidden layers beneath famous paintings.  
These hidden layers may come from:
1. Canvas reuse (older sketches/paintings under new works)
2. Later overpainting to conceal details
3. Artist revisions during composition
4. Restoration interventions

This project applies AI to high-quality multispectral data to automate analysis, identify painting properties, and detect hidden imagery.

### Project Information
1. **Total duration:** 175 hours
2. **Difficulty:** Medium
3. **Corresponding project:** ArtExtract
4. **Participating organization:** University of Alabama

### Task Ideas
1. Extract useful information from multispectral images:
   - pigment-related cues
   - damage patterns
   - restoration traces
2. Determine whether a painting contains a hidden painted layer.

### Expected Results
1. Prepare a multispectral painting dataset.
2. Train a model to identify painting properties.
3. Train a model to detect hidden images beneath paintings.
4. Optional: train a model to reconstruct/retrieve hidden imagery.

### Requirements
1. Python
2. PyTorch or TensorFlow
3. Computer vision experience

### Test
Use the project test link provided by HumanAI.  
If you publish this page, replace this line with the exact URL.

### Mentors
1. Emanuele Usai (University of Alabama)
2. Sergei Gleyzer (University of Alabama)

Do **not** contact mentors directly by email.  
Send communication and submissions to **human-ai@cern.ch** with project title, CV, and test results.

---

## Applicant Submission Instructions (GSoC Evaluation)

1. Work in your own GitHub branch/repository.
2. Do **not** open PRs to the project repository for the evaluation test.
3. Submit your materials at least 1 week before the GSoC proposal deadline (earlier preferred).

Email: **human-ai@cern.ch**  
Subject: **Evaluation Test: ArtExtract**

Include:
1. CV
2. Link to repository
3. Jupyter notebook
4. PDF export of notebook with outputs

---

## Tasks for Prospective GSoC 2025 Applicants

Below are the evaluation tasks for ArtExtract applicants.

### General Rules
1. Work in your own GitHub branch/repository.
2. Do **not** submit PRs to the project repository for this test.
3. Share the final GitHub link when complete.
4. Submit at least 1 week before the proposal deadline when possible.

### Test Submission Instructions
Send to **human-ai@cern.ch** with subject: **Evaluation Test: ArtExtract**.

Required submission package:
1. CV
2. Link to all completed code/work
3. Jupyter notebook
4. PDF of notebook with outputs

### Task 1: Multi-Task Painting Classification Pipeline
Build a convolutional-recurrent model for classifying:
1. Style
2. Artist
3. Genre
4. Additional general/specific attributes where available

Dataset:
- [ArtGAN WikiArt Dataset README](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md)

Required in your report:
1. Chosen architecture and strategy, with rationale.
2. Evaluation metrics used and why they fit the task.
3. Outlier analysis:
   - paintings that do not fit assigned artist/genre labels
   - possible causes (label noise, atypical style, transition cases)
