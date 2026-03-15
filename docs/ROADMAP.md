# Roadmap

This document is the baseline for future development planning.

## Immediate Priorities

### Task 1

1. Standardize manifest generation across style, artist, and genre splits.
2. Improve evaluation beyond top-k accuracy with macro-F1, calibration, and per-class confusion analysis.
3. Make outlier scoring reproducible and traceable to logits or embedding distances.

### Task 2

1. Stabilize first-run dependency and model download behavior.
2. Add repeatable retrieval evaluation datasets and metrics exports.
3. Support metadata-aware filtering for artist, period, or object type constraints.

### Multispectral Track

1. Replace synthetic-only assumptions with a stricter real-data manifest contract.
2. Expand property labeling support for pigment, damage, and restoration targets.
3. Tighten reconstruction evaluation and example export quality.

## Engineering Backlog

1. unify output schemas across task entrypoints
2. add stronger CLI validation and clearer failure messages
3. increase automated coverage for Task 1 and Task 2
4. move generated run outputs away from versioned snapshots toward reproducible commands
5. document dataset preparation scripts with input and output contracts

## Acceptance Standard For New Work

Any future feature should satisfy all of the following:

1. executable from `scripts/`
2. documented in one of the primary Markdown files
3. reproducible from repository state without private assumptions
4. covered by at least a smoke test or deterministic verification path

## Deferred Work

1. stronger multimodal fusion for multispectral and X-ray channels
2. retrieval-assisted hidden-image reconstruction
3. curator-facing review tooling for outliers and hidden detections
