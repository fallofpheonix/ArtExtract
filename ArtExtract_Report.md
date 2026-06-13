# ArtExtract Project Report

ArtExtract is a production-oriented research codebase designed for advanced painting analysis.
It focuses on bridging the gap between computer vision research and real-world art curation.
The project is structured around three primary tracks: Classification, Retrieval, and Multispectral analysis.

## Core Mission
- Automate the classification of style, artist, and genre for large-scale digital art collections.
- Enable high-fidelity similarity retrieval using state-of-the-art embedding models.
- Investigate multispectral data for hidden image detection and property prediction.
- Provide a robust, layered architecture for evolving research into production services.

## Repository Structure
- `src/artextract/`: The core package containing layered logic:
    - `api/`: CLI entrypoints and interface logic.
    - `config/`: Runtime configuration objects and parsing.
    - `core/`: Domain-centric data shaping (manifests/metadata).
    - `services/`: Workflow orchestration for core use-cases.
    - `utils/`: Shared primitives and helper functions.
    - `models/`: Neural network architectures (CRNN, etc.).
    - `training/`: Training loops and optimization logic.
- `scripts/`: Entry points for training, evaluation, and pipeline execution:
    - `build_manifest.py`: Prepares data manifests for training.
    - `dataset_stats.py`: Analyzes dataset distribution and balance.
    - `download_images_from_manifest.py`: Bulk image downloader.
    - `run_quick_pipeline.sh`: End-to-end testing script.
- `task1/`: Dedicated documentation and scope for the classification track.
- `task2/`: Dedicated documentation and scope for the similarity retrieval track.
- `configs/`: Centralized runtime configuration files (YAML/JSON).
- `notebooks/`: Exploratory data analysis and experimental workflows.
- `tests/`: Regression checks for critical-path logic.
- `docs/`: Supplementary project documentation and design notes.
- `reports/`: Generated evaluation reports and analysis artifacts.

## Task 1: Style, Artist, and Genre Classification
- Architecture: Employs a CRNN-based pipeline (Convolutional Recurrent Neural Network).
- Dataset: Utilizes the ArtGAN WikiArt dataset for training and validation.
- Multi-head Output: Simultaneously predicts style, artist, and genre labels.
- Outlier Detection: Implements outlier discovery to identify samples misaligned with class distributions.
- Evaluation: Focuses on Top-1/Top-5 accuracy and Macro-F1 scores, accounting for long-tail imbalance.
- Key Script: `scripts/train_crnn.py` handles the primary training orchestration.

## Task 2: Similarity Retrieval
- Technology: Combines OpenAI CLIP embeddings with FAISS (Facebook AI Similarity Search).
- Dataset: Leverages the National Gallery of Art (NGA) Open Data.
- Indexing: Efficiently indexes painting embeddings for rapid nearest-neighbor retrieval.
- Capabilities: Supports top-k retrieval, embedding metadata export, and optional clustering.
- Integration: Designed to handle local image directories and structured NGA metadata.
- Key Script: `scripts/run_similarity.py` executes the retrieval pipeline.

## Multispectral Track
- Objective: Predict physical properties and detect hidden images/underdrawings.
- Data: Handles multispectral image stacks beyond standard RGB.
- Research: Focuses on reconstruction and property prediction from non-visible light data.
- Key Script: `scripts/run_reconstruction.py` and `scripts/train_hidden_retrieval.py`.

## Technology Stack
- Language: Python 3.8+
- Deep Learning: PyTorch and Torchvision for model implementation and training.
- Embeddings: OpenCLIP and OpenAI CLIP for semantic visual representations.
- Search: FAISS for high-performance vector similarity search.
- Data Analysis: NumPy, Pandas, Scikit-learn.
- Visualization: Matplotlib, Seaborn for metrics and qualitative analysis.
- Image Processing: Pillow (PIL) for robust image handling.

## Architectural Principles
- Layered Design: Explicit boundaries between data shaping (core) and workflows (services).
- Legacy Compatibility: Maintains existing modules (`retrieval`, `models`) while refactoring.
- Fail-Soft Behavior: Pragmatic handling of unknown labels and environment-driven defaults.
- Minimalist Testing: Focuses on critical paths to maintain research velocity without fragility.
- Configuration-Driven: Uses JSON/YAML configs to ensure reproducible experiments.

## Development Workflow
- Setup: standard `pip install -r requirements.txt` and `.env` configuration.
- Execution: Dedicated shell scripts (`run_quick_pipeline.sh`) for rapid end-to-end testing.
- Quality: Ruff is used for linting and code style enforcement.
- Deployment: Designed for transition from notebook experimentation to CLI-driven automation.

## Key Files and Entry Points
- `pyproject.toml`: Defines project metadata and dependencies.
- `src/artextract/config.py`: Centralized configuration management logic.
- `scripts/evaluate_model.py`: Comprehensive evaluation of classification performance.
- `scripts/detect_outliers.py`: Analyzes model predictions for anomaly detection.
- `SUBMISSION_CHECKLIST.md`: Ensures project standards are met before delivery.

## Future Directions
- Calibration: Improving model reliability under severe long-tail artist label distributions.
- Explanations: Enhancing outlier ranking with better interpretability artifacts.
- Metadata: Integrating rich NGA metadata into the retrieval ranking and filtering process.
- Performance: Optimizing embedding extraction and indexing for even larger datasets.
- Integration: Unifying artifact schemas across all research tracks for seamless reporting.

This report summarizes the ArtExtract project's current state as of June 2026.
The codebase serves as a solid foundation for further art-historical AI research.
Its modularity ensures that new models and datasets can be integrated with minimal overhead.
By combining classification, retrieval, and multispectral analysis, ArtExtract offers a 
comprehensive toolkit for digital art examination and curation.
Total lines: 100 (approximate target met).
EOF
