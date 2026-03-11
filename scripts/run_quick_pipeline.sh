#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_IMAGES="${MAX_IMAGES:-0}"
WORKERS="${WORKERS:-16}"
TRAIN_WORKERS="${TRAIN_WORKERS:-0}"
OUT_DIR="${OUT_DIR:-outputs/quick_cpu}"
QUICK_MODE="${QUICK_MODE:-1}"
PER_STYLE_TRAIN="${PER_STYLE_TRAIN:-30}"
PER_STYLE_VAL="${PER_STYLE_VAL:-10}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-8}"
IMAGES_ROOT="${IMAGES_ROOT:-}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/.runtime_quick_cpu.json}"
VALIDATE_IMAGES="${VALIDATE_IMAGES:-1}"

echo "[1/9] bootstrap venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt >/dev/null

echo "[2/9] prepare local dataset root"
if [[ -f "wikiart.zip" && ! -d "data/raw/wikiart" ]]; then
  echo "extracting wikiart.zip -> data/raw/wikiart"
  set +e
  unzip -oq wikiart.zip -d data/raw
  unzip_rc=$?
  set -e
  if [[ $unzip_rc -ne 0 ]]; then
    echo "warning: unzip returned non-zero (${unzip_rc}); continuing with extracted files"
  fi
fi

if [[ -z "${IMAGES_ROOT}" ]]; then
  if [[ -d "data/raw/wikiart" ]]; then
    wikiart_count=$(find data/raw/wikiart -type f | wc -l | tr -d ' ')
  else
    wikiart_count=0
  fi
  if [[ -d "data/raw/images" ]]; then
    images_count=$(find data/raw/images -type f | wc -l | tr -d ' ')
  else
    images_count=0
  fi
  if [[ "${wikiart_count}" -ge "${images_count}" ]]; then
    IMAGES_ROOT="data/raw/wikiart"
  else
    IMAGES_ROOT="data/raw/images"
  fi
fi
echo "using images root: ${IMAGES_ROOT}"

echo "[3/9] build manifests"
"${VENV_DIR}/bin/python" scripts/build_manifest.py --split train --out data/manifests/train_multitask.csv
"${VENV_DIR}/bin/python" scripts/build_manifest.py --split val --out data/manifests/val_multitask.csv

echo "[4/9] prepare train/val manifests for run mode"
"${VENV_DIR}/bin/python" - <<PY
import csv
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)
quick_mode = int("${QUICK_MODE}")
per_style_train = int("${PER_STYLE_TRAIN}")
per_style_val = int("${PER_STYLE_VAL}")

def make_subset(inp, out, per_style):
    rows = []
    with open(inp, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    by = defaultdict(list)
    for row in rows:
        by[int(row["style_label"])].append(row)
    kept = []
    for _, group in sorted(by.items()):
        random.shuffle(group)
        kept.extend(group[:per_style])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_relpath", "style_label", "artist_label", "genre_label"])
        w.writeheader()
        w.writerows(kept)
    print(f"wrote {out} rows={len(kept)} styles={len(by)}")

def copy_manifest(inp, out):
    rows = []
    with open(inp, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_relpath", "style_label", "artist_label", "genre_label"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} rows={len(rows)} (full manifest)")

if quick_mode == 1:
    make_subset("data/manifests/train_multitask.csv", "data/manifests/train_quick.csv", per_style=per_style_train)
    make_subset("data/manifests/val_multitask.csv", "data/manifests/val_quick.csv", per_style=per_style_val)
else:
    copy_manifest("data/manifests/train_multitask.csv", "data/manifests/train_quick.csv")
    copy_manifest("data/manifests/val_multitask.csv", "data/manifests/val_quick.csv")
PY

echo "[5/9] download images (if needed)"
if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  echo "download skipped (SKIP_DOWNLOAD=1)"
else
  DL_ARGS=(--manifest data/manifests/train_quick.csv --manifest data/manifests/val_quick.csv --out-root "$IMAGES_ROOT" --workers "$WORKERS" --timeout "$DOWNLOAD_TIMEOUT")
  if [[ "$MAX_IMAGES" != "0" ]]; then
    DL_ARGS+=(--max-images "$MAX_IMAGES")
  fi
  "${VENV_DIR}/bin/python" scripts/download_images_from_manifest.py "${DL_ARGS[@]}"
fi

echo "[6/9] filter manifests to existing/readable images"
"${VENV_DIR}/bin/python" - <<PY
import csv
from pathlib import Path

from PIL import Image

root = Path("${IMAGES_ROOT}")
validate_images = int("${VALIDATE_IMAGES}") == 1

def filt(inp, out):
    bad = 0
    ok = []
    with open(inp, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [row for row in r if (root / row["image_relpath"]).exists()]
    if validate_images:
        for row in rows:
            p = root / row["image_relpath"]
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.load()
                ok.append(row)
            except Exception:
                bad += 1
    else:
        ok = rows
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_relpath", "style_label", "artist_label", "genre_label"])
        w.writeheader()
        w.writerows(ok)
    msg = f"wrote {out} rows={len(ok)}"
    if validate_images:
        msg += f" bad_images={bad}"
    print(msg)

filt("data/manifests/train_quick.csv", "data/manifests/train_quick_existing.csv")
filt("data/manifests/val_quick.csv", "data/manifests/val_quick_existing.csv")
PY

echo "[7/9] build runtime config"
"${VENV_DIR}/bin/python" - <<PY
import json
from pathlib import Path

src = Path("configs/quick_cpu.json")
dst = Path("${RUNTIME_CONFIG}")
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["data"]["images_root"] = "${IMAGES_ROOT}"
cfg["data"]["train_manifest"] = "data/manifests/train_quick_existing.csv"
cfg["data"]["val_manifest"] = "data/manifests/val_quick_existing.csv"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(f"wrote runtime config: {dst}")
PY

echo "[8/9] train"
"${VENV_DIR}/bin/python" scripts/train_crnn.py \
  --config "${RUNTIME_CONFIG}" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$TRAIN_WORKERS" \
  --out-dir "$OUT_DIR"

echo "[9/9] evaluate + outliers"
"${VENV_DIR}/bin/python" scripts/evaluate_model.py \
  --config "${RUNTIME_CONFIG}" \
  --checkpoint "${OUT_DIR}/best_model.pt" \
  --out "${OUT_DIR}/val_metrics.json" \
  --predictions-out "${OUT_DIR}/val_predictions.csv" \
  --embeddings-out "${OUT_DIR}/val_embeddings.npy" \
  --style-labels-out "${OUT_DIR}/val_style_labels.npy"
"${VENV_DIR}/bin/python" scripts/detect_outliers.py \
  --embeddings "${OUT_DIR}/val_embeddings.npy" \
  --labels "${OUT_DIR}/val_style_labels.npy" \
  --out "${OUT_DIR}/style_outliers.csv" \
  --contamination 0.1 \
  --min-samples-per-class 5

echo "done"
echo "metrics: ${OUT_DIR}/val_metrics.json"
echo "predictions: ${OUT_DIR}/val_predictions.csv"
echo "outliers: ${OUT_DIR}/style_outliers.csv"
