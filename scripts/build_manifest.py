#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artextract.data import build_multitask_rows
from artextract.utils import ensure_parent_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Build merged multi-task manifest from split CSVs")
    p.add_argument("--metadata-dir", default="data/metadata/wikiart_csv")
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = build_multitask_rows(args.metadata_dir, args.split)
    ensure_parent_dir(args.out)

    with Path(args.out).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_relpath", "style_label", "artist_label", "genre_label"])
        for r in rows:
            w.writerow([r.image_relpath, r.style_label, r.artist_label, r.genre_label])

    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
