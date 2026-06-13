# ArtExtract Action Plan (Revised)

## Week 1: Critical Fixes (COMPLETED)
- [x] Delete or Merge legacy modules (`retrieval/`, `models/`, `training/`, `reconstruction/`)
- [x] Modify CLI arguments + improve `--help`
- [x] Add Dockerfile (multi-stage)
- [x] Unify config format (YAML preferred)
- [x] Add multi-threading to `download_images_from_manifest.py`

## Month 2: Scientific Reliability
- [x] **Define Artifact Model**: Establish IDs, owners, and versioning for the full research graph.
- [ ] **Dataset Versioning**: Integrate DVC for manifest and image snapshot tracking.
- [ ] **Experiment Registry**: Set up MLflow for tracking hyperparams, metrics, and model weights.
- [ ] **Benchmark Suite**: Create a "Gold Standard" test set and automated regression reports.
- [ ] **Dataset Validation**: Implement automated checks for duplicates, corruption, and label quality.
- [ ] **Reproducibility**: Add random seed tracking and dataset hashing to all training scripts.

## Month 3: Research Excellence
- [ ] **ViT Support**: Add Vision Transformer backends for better feature extraction.
- [ ] **CLIP Fine-tuning**: Implement fine-tuning on art-specific datasets (e.g., WikiArt).
- [ ] **Grad-CAM**: Add attention/saliency visualization for model interpretability.
- [ ] **Uncertainty Estimation**: Add Bayesian layers or entropy scores to classification outputs.
- [ ] **Active Learning**: Build a CLI-based prototype for iterative curator feedback.

## Month 4: Production Readiness
- [ ] **FastAPI REST API**: Build an API layer for model inference and status tracking.
- [ ] **Metadata Database**: Migrate from YAML manifests to PostgreSQL for 1M+ image scale.
- [ ] **Object Storage**: Add support for S3-compatible backends for image storage.
- [ ] **Quantization**: Implement INT8/ONNX export for faster, lightweight inference.
- [ ] **Monitoring**: Add health checks and OpenTelemetry tracing.

## Month 5: Scalability & Performance
- [ ] **Redis Caching**: Cache embeddings and frequent retrieval results.
- [ ] **Performance Profiling**: Conduct a deep dive into inference and data loading bottlenecks.
- [ ] **Incremental Indexing**: Support FAISS index updates without full rebuilds.
- [ ] **Horizontal Scaling**: Optimize the pipeline for multi-GPU or distributed environments.

## Month 6: Usability & Community
- [ ] **Web UI**: Build a React/Next.js dashboard for visualizing search and classification results.
- [ ] **Demo Deployment**: Deploy a public demo with a curated artwork sample.
- [ ] **CONTRIBUTING.md / CODE_OF_CONDUCT.md**: Finalize community guidelines.
- [ ] **Benchmark Dashboard**: Publish a public dashboard for tracking model improvements.

---

## Detailed Implementation Notes (Month 2 Focus)
- **DVC**: Essential for ensuring that research results are tied to specific data versions.
- **MLflow**: Replaces the manual `run_meta.yaml` tracking with a robust system.
- **Benchmarks**: Prevents the "architectural sprawl" from degrading model performance.
- **Validation**: High-quality data is the foundation of high-quality research.
