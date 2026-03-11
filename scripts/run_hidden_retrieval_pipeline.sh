#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-12}"
MAX_IMAGES="${MAX_IMAGES:-2500}"
OUT_DIR="${OUT_DIR:-outputs/hidden_retrieval_quick}"
TRAIN_WORKERS="${TRAIN_WORKERS:-0}"
CONFIG_PATH="${CONFIG_PATH:-configs/retrieval_baseline.json}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/.runtime_retrieval.json}"
IMAGES_ROOT="${IMAGES_ROOT:-}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt >/dev/null

if [[ -f "wikiart.zip" && ! -d "data/raw/wikiart" ]]; then
  set +e
  unzip -oq wikiart.zip -d data/raw
  unzip_rc=$?
  set -e
  if [[ $unzip_rc -ne 0 ]]; then
    echo "warning: unzip returned ${unzip_rc}; continuing"
  fi
fi

if [[ -z "${IMAGES_ROOT}" ]]; then
  if [[ -d "data/raw/wikiart" ]]; then
    IMAGES_ROOT="data/raw/wikiart"
  else
    IMAGES_ROOT="data/raw/images"
  fi
fi

echo "using retrieval images root: ${IMAGES_ROOT}"

"${VENV_DIR}/bin/python" - <<PY
import json
from pathlib import Path

src = Path("${CONFIG_PATH}")
dst = Path("${RUNTIME_CONFIG}")
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["data"]["images_root"] = "${IMAGES_ROOT}"
cfg["data"]["max_images"] = int("${MAX_IMAGES}")
cfg["training"]["epochs"] = int("${EPOCHS}")
cfg["training"]["batch_size"] = int("${BATCH_SIZE}")
cfg["training"]["num_workers"] = int("${TRAIN_WORKERS}")
cfg["training"]["out_dir"] = "${OUT_DIR}"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(f"wrote runtime config: {dst}")
PY

"${VENV_DIR}/bin/python" scripts/train_hidden_retrieval.py \
  --config "${RUNTIME_CONFIG}" \
  --out-dir "${OUT_DIR}"

"${VENV_DIR}/bin/python" scripts/evaluate_hidden_retrieval.py \
  --config "${RUNTIME_CONFIG}" \
  --checkpoint "${OUT_DIR}/best_model.pt" \
  --out "${OUT_DIR}/val_metrics_eval.json"

echo "done"
echo "train metrics: ${OUT_DIR}/metrics.json"
echo "eval metrics: ${OUT_DIR}/val_metrics_eval.json"
echo "preview: ${OUT_DIR}/val_preview.png"
