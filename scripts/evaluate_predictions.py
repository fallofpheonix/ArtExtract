#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artextract.training.metrics import classification_metrics


def _load(path: str):
    yt, yp = [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yt.append(int(row["y_true"]))
            yp.append(int(row["y_pred"]))
    return yt, yp


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate classification predictions")
    p.add_argument("--predictions", required=True, help="CSV with columns y_true,y_pred")
    args = p.parse_args()

    yt, yp = _load(args.predictions)
    m = classification_metrics(yt, yp)
    print(json.dumps(m, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
