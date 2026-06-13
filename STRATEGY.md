# ArtExtract Strategy & Vision

This document defines the core identity, technical architecture, and long-term roadmap for ArtExtract, answering the fundamental questions required to move from research to a stable platform.

## 1. Product & Scope

| Question | Answer |
|----------|--------|
| **1. Primary User** | **Researchers** (Art Historians, Conservators, and AI Researchers). |
| **2. Most Important Feature** | **Multispectral analysis** (specifically hidden image detection and property prediction). |
| **3. Project Type** | **Research Platform**. It prioritizes flexibility and accuracy over consumer-grade production scale. |
| **4. Problem Solved** | Bridges the gap between high-end Computer Vision (CLIP/FAISS) and the specialized needs of multispectral art examination. |
| **5. Expected User Scale** | **100s** (Institutional and academic users). Not intended for mass-market consumption. |
| **6. Offline-Capable?** | **Yes.** Essential for handling high-resolution, sensitive cultural heritage data in secure environments. |
| **7. Commercial Goal?** | **No.** The focus is on cultural heritage, provenance research, and academic advancement. |
| **8. API Priority** | **Secondary.** CLI and Python Notebook integration are the primary interfaces for the research workflow. |
| **9. Success Metric** | **Empirical discovery** (e.g., finding a hidden underdrawing) and **Classification F1-score on long-tail artist data**. |
| **10. Feature to Remove** | **Legacy non-layered modules.** (Deprecated in favor of the `core`/`services` structure). |

---

## 2. Architecture

| Question | Answer |
|----------|--------|
| **11. Monolith or Microservices?** | **Modular Monolith.** Microservices are premature for the current user scale and complexity. |
| **12. Separate Services?** | No. They should be **modular packages** (`core.models`, `services.retrieval`) within a single execution environment. |
| **13. Bottlenecks** | **Disk I/O** (loading high-res TIFFs) and **CPU/RAM** (pre-processing image stacks for CLIP). |
| **14. State Storage** | **Filesystem.** Manifests (CSV/YAML), embeddings (FAISS/NumPy), and weights (PyTorch). |
| **15. Stateless Components** | Inference logic, pre-processing functions, and evaluation metrics. |
| **16. Config Versioning** | **Git.** Stored in the `configs/` directory as part of the repository. |
| **17. Model Tracking** | **Artifact Metadata.** Each training run generates a `run_meta.yaml` and `training_history.yaml`. |
| **18. Deployment Target** | **Local Workstations** and **Institutional GPU Clusters**. |
| **19. Wrong Decision?** | Potential **over-layering**. The current `core/services` split might be too rigid for rapid research iteration. |
| **20. High Maintenance** | **Multispectral data loaders.** Handling varied sensor data and alignment is complex and prone to breaking. |

---

## 3. AI / ML

| Question | Answer |
|----------|--------|
| **21. Most Important Task** | **Similarity (Retrieval)** and **Multispectral Reconstruction**. |
| **22. Current Accuracy** | ~75-80% on style classification (top-1); varies heavily by artist label distribution. |
| **23. Required Accuracy** | **High Precision** (>90%) for similarity retrieval to be useful for curators. |
| **24. Label Validation** | Currently relies on **WikiArt/NGA metadata**. Future: Curator-in-the-loop validation. |
| **25. Label Noise** | **Medium to High.** Common in large-scale art datasets where attributions change. |
| **26. CRNN vs. ViT** | CRNN was the baseline; **ViT is the priority upgrade** for better global/local feature integration. |
| **27. CLIP vs. DINOv2** | CLIP is used for its semantic (text-image) power; DINOv2 is a candidate for pure visual property prediction. |
| **28. Embeddings Status** | **Frozen** (pre-trained). Fine-tuning on art-specific datasets is a Tier 2 priority. |
| **29. Model Drift** | Measured via **Regression Tests** on a static benchmark set of 100 "gold standard" paintings. |
| **30. Uncertainty Estimation** | **Not currently implemented.** Priority for Tier 2 to help curators trust model outputs. |
| **31. False Positives** | Analyzed via **Grad-CAM** to see if the model is over-focusing on frames or background textures. |
| **32. False Negatives** | Typically due to **resolution loss** or **atypical artist periods**. |
| **33. Resource Consumer** | **CLIP Embedding Extraction** (Inference) and **Multispectral Training**. |
| **34. acceptable Training** | **< 24 hours** on a single A100/3090 for the full WikiArt set. |
| **35. Largest Dataset (3y)** | **~1 Million images** (e.g., combining NGA, MET, and Rijksmuseum collections). |

---

## 4. Data

| Question | Answer |
|----------|--------|
| **36. Datasets Supported** | **WikiArt** (Classification) and **NGA Open Data** (Retrieval). |
| **37. Current Image Count** | ~80k (WikiArt subset) + ~2k (NGA sample). |
| **38. Avg Image Size** | **3-5 MB** (Compressed RGB); **100+ MB** (Multispectral stacks). |
| **39. Duplicate Detection** | **Perceptual Hashing** (not yet implemented; priority for Tier 1 cleanup). |
| **40. Dataset Versioning** | **None.** (Priority: Move to **DVC** or simple hash-based manifest versioning). |
| **41. Corrupt Images** | Detected during **Image.open()** in the data loader with fail-soft logging. |
| **42. Mandatory Metadata** | `image_path`, `artist`, `style`. |
| **43. Missing Metadata** | Labeled as **'unknown'** to ensure pipeline continuity. |
| **44. Annotation Storage** | **CSV/YAML Manifests.** |
| **45. Max Dataset Scale** | **1 Million paintings.** |
| **46. Multispectral %** | **< 5%.** It is a high-value but low-volume data track. |
| **47. Update Frequency** | **Ad-hoc.** Driven by new museum data releases. |
| **48. Data Immutability** | **No.** Labels are often updated; the raw imagery is considered immutable. |
| **49. Data Governance** | **License-driven.** Strictly follows `DATA_RIGHTS.md` for redistribution rules. |
| **50. Dataset Removal** | The system must be able to **re-index** from manifests if a source disappears. |

---

## 5. Retrieval & Search

| Question | Answer |
|----------|--------|
| **51. Why FAISS?** | Industry standard for **speed** and **memory efficiency** at the scale of 10k-1M vectors. |
| **52. Index Type** | **Flat (L2)** for small sets; **IVF-Flat** for larger collections. |
| **53. Indexing Time** | **~30 mins** for 100k images on a modern CPU/GPU. |
| **54. Rebuild Frequency** | Whenever the embedding model changes or the dataset grows by >10%. |
| **55. Incremental Updates** | Supported by FAISS, but **rebuilding** is preferred for small art collections to maintain precision. |
| **56. Latency Target** | **< 200ms** for a single image query. |
| **57. Evaluation** | **Top-k Precision/Recall** based on genre/artist clusters. |
| **58. Bad Result** | A visually identical painting of a different style/period (semantic failure). |
| **59. Metadata Filtering?** | **Yes.** Essential (e.g., "Find similar paintings *only* from the 17th century"). |
| **60. Hybrid Search?** | **Yes.** Vector search for visuals + SQL/Metadata for filtering. |

---

## 6. Multispectral Track

| Question | Answer |
|----------|--------|
| **61. Core Feature?** | **Core Research Experiment.** It's the most innovative but least "stable" part of the project. |
| **62. Targeted Sensors** | RGB, Infrared (IR), Ultraviolet (UV), and X-Ray. |
| **63. Spectral Bands** | Currently supports 4-band stacks (RGB + 1 modality) or 6-band stacks. |
| **64. Reconstruction Metrics** | **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index). |
| **65. Ground Truths** | High-res institutional scans of known underdrawings (often manual/curated). |
| **66. Unique Value** | Non-invasive discovery of hidden art-historical context (e.g., changes in composition). |
| **67. Compute Required** | **High.** Requires significant VRAM for multi-channel convolutional training. |
| **68. Consumer Hardware?** | **Yes (Inference)**; training is recommended on high-end desktop GPUs (8GB+ VRAM). |
| **69. Failure Modes** | Misalignment of spectral bands (registration errors) leading to blurred reconstructions. |
| **70. Separate Project?** | **No.** The synergy between RGB classification and multispectral analysis is a key strength. |

---

## 7. Performance & Scaling

| Question | Answer |
|----------|--------|
| **71. RAM Requirements** | 8GB minimum; 16GB+ recommended for large FAISS indices. |
| **72. GPU Requirements** | 4GB VRAM for inference; 8GB+ for training. |
| **73. Performance Bottleneck** | **Preprocessing.** Resizing and normalizing high-res images for model ingestion. |
| **74. Max FAISS Index** | **~1 Million vectors.** Beyond this, IVFPQ or HNSW would be required. |
| **75. Inference Latency** | ~50ms for embedding extraction; ~10ms for vector search. |
| **76. Training Latency** | ~5-10 mins per epoch on WikiArt (subset) with a modern GPU. |
| **77. Storage Required** | ~500GB for a comprehensive 100k-image high-res collection. |
| **78. Resource Exhaustion** | Fail-soft behavior; batch size auto-scaling or graceful termination. |
| **79. Disaster Recovery** | Reproducible builds from `pyproject.toml` and data manifests (Git-tracked). |
| **80. First Scaling Issue** | **Memory** exhaustion when loading too many high-res images into the DataLoader queue. |

---

## 8. Open Source Strategy

| Question | Answer |
|----------|--------|
| **81. Why Join?** | To advance the state of the art in digital humanities and museum AI. |
| **82. Needed Contributions** | **Dataset connectors**, **Alignment algorithms** for multispectral data, and **UI/UX design**. |
| **83. Missing Docs** | Advanced researcher's guide for fine-tuning on custom datasets. |
| **84. Onboarding Friction** | Environment setup (FAISS/Torch) and data acquisition. |
| **85. % Tested** | **~40%** (Critical paths). Goal is 70% coverage for Tier 1 components. |
| **86. Breaking Changes** | Communicated via **GitHub Releases** and updated `TASKS.md`. |
| **87. Governance Model** | **Benevolent Dictator for Life (BDFL)** (Maintainer-led research). |
| **88. Release Management** | Semantic versioning; automated CI/CD for package builds. |
| **89. Benchmark Publishing** | Weekly/Monthly automated reports in the `reports/` directory. |
| **90. Growth Barrier** | Specialized nature of the data and lack of a non-technical GUI. |

---

## 9. Future Vision

| Question | Answer |
|----------|--------|
| **91. 1-Year Goal** | A stable, production-ready research toolkit with a basic Web UI. |
| **92. 3-Year Goal** | The industry-standard open-source platform for museum collection analysis. |
| **93. Researcher Attraction** | State-of-the-art multispectral and classification metrics. |
| **94. Museum Attraction** | Easy-to-use search and visualization for their private digital archives. |
| **95. Investor Attraction** | (N/A) - Focus is on grant funding and institutional partnerships. |
| **96. Technical Risk** | Complexity of multispectral data registration and alignment. |
| **97. Most Expensive** | **Data acquisition and cleaning.** |
| **98. Most Defensible** | **Domain-specific models** trained on curated, high-quality museum data. |
| **99. Post-Funding** | The open-source codebase and the research artifacts/benchmarks remain. |
| **100. Mission Statement** | **"Empowering art discovery through accessible, high-performance AI and multispectral analysis."** |

---

## 10. Adversarial Reality Check & Constraints

This section acknowledges the risks of architectural sprawl and defines the boundaries of the ArtExtract research platform.

### Product Reality Check
- **Multispectral Focus**: While only 5% of data is multispectral, it represents the **unique scientific value** of the platform. RGB classification is a commodity; multispectral discovery is a differentiator.
- **UI vs. Tooling**: **Correction:** Research tooling (DVC, Experiment tracking) MUST precede the React UI. A UI for a scientifically unreliable model is counter-productive.
- **Trust & Uncertainty**: Curators should NOT trust AI outputs blindly. Tier 2 priority is to implement **uncertainty estimation** (Bayesian layers or entropy scores) to flag low-confidence results.
- **Measurable Outcome**: The first outcome is a **verified match** between a hidden underdrawing and a known historical sketch.

### Architecture Challenges
- **Filesystem Scale**: The filesystem is a bottleneck at 1M+ images. A migration strategy to an **Object Store (S3-compatible)** and a **SQL Metadata DB** is planned for Month 4 (Production Readiness).
- **Concurrency**: Manifests are currently Git-tracked. For multi-researcher environments, a **Centralized Experiment Registry (MLflow)** is required to prevent configuration drift.
- **Monolith Bottleneck**: The monolith will remain until **inference latency** or **training data throughput** exceeds the capacity of single-node GPU workstations.

---

## 12. Artifact Model & Lifecycle

Before implementing DVC or MLflow, the project defines the following canonical artifact graph. Every node must be traceable, versioned, and validated.

| Artifact | ID Strategy | Owner | Storage | Versioning |
|----------|-------------|-------|---------|------------|
| **Dataset** | `ds_<manifest_hash>` | DVC | S3-compatible | Git + DVC |
| **Preprocessing** | `pp_<config_hash>` | Git | `configs/` | Git Commit |
| **Experiment** | `run_<uuid>` | MLflow | SQLite / Files | MLflow Run ID |
| **Model** | `mod_<run_id>` | MLflow | MLflow Registry | SemVer (v1.0.0) |
| **Embeddings** | `emb_<mod_id>_<ds_id>` | Filesystem | `data/embeddings/` | Hash(Model + Data) |
| **Index** | `idx_<emb_id>_<type>` | Filesystem | `data/indices/` | Hash(Embeddings) |
| **Benchmark** | `bench_<run_id>_<suite_v>` | Git / MLflow | `reports/` | Linked to Run ID |

---

## 13. Research Reliability Standard (Questions & Answers)

### Dataset Versioning
- **Definition**: A dataset version is a **Manifest (CSV)** + **Image Files (DVC)** + **Split Logic**.
- **Change Trigger**: Any modification to a manifest row or an image file content.
- **Lineage**: Manifests include a `source_dataset` field to track origins (e.g., NGA-v1).
- **Simultaneity**: Supported via Git branches and DVC checkouts; multiple researchers can work on different dataset versions in parallel.
- **Copyright**: DVC-remote access is restricted; the "redistributable" flag in the manifest filters public snapshots.

### Experiment & MLflow
- **Hierarchy**: **Experiment** (e.g., style-classification) -> **Run** (Individual execution).
- **Mandatory Metadata**: Git commit, `ds_hash`, `pp_hash`, CPU/GPU specs, Python environment.
- **Artifacts**: Every run MUST store `config.yaml`, `requirements.txt`, and the `best_model.pt`.
- **Promotion**: A model is promoted to 'Production' or 'Staging' only after passing the **Gold Standard Benchmark Suite**.
- **Approval**: Required from the lead maintainer via MLflow Model Registry transitions.

### Benchmark Suite
- **Gold Standards**: 
    - **Retrieval**: 500 NGA images with expert-verified similarity clusters.
    - **Classification**: 1000 WikiArt images with high-confidence, expert-validated labels.
- **Regression Policy**: Any drop >1% in 'Gold' metrics blocks a PR merge.
- **Verifiability**: Labels are curated by project historians; the "Gold" dataset itself is immutable and tracked in DVC as `ds_gold_v1`.

### Research Integrity & Traceability
- **Reproducibility**: Random seeds are logged, but **Environment Snapshots** (Docker image ID + Pip freeze) are the primary truth for execution context.
- **Canonical Source of Truth**:
    - **Code/Config**: Git.
    - **Data/Images**: DVC.
    - **Experiments/Models**: MLflow.
    - **Final Reports**: Git (in `reports/`).

---

## 14. Artifact Lifecycle
```text
Dataset (DVC)
   ↓
Preprocessing (Git/Config)
   ↓
Experiment (MLflow Tracking)
   ↓
Model (MLflow Registry)
   ↓
Embedding Set (Filesystem/Object Store)
   ↓
FAISS Index (Filesystem/Object Store)
   ↓
Benchmark Result (Git/Reports)
```
Every transition is recorded in the MLflow **Run Metadata**.
