#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter


def main() -> int:
    p = argparse.ArgumentParser(description="Compute manifest label distribution stats")
    p.add_argument("--manifest", required=True)
    args = p.parse_args()

    style = Counter()
    artist = Counter()
    genre = Counter()
    n = 0

    with open(args.manifest, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n += 1
            style[int(row["style_label"])] += 1
            artist[int(row["artist_label"])] += 1
            genre[int(row["genre_label"])] += 1

    print(f"samples={n}")
    print(f"unique_style={len(style)} min={min(style.values()) if style else 0} max={max(style.values()) if style else 0}")
    print(f"unique_artist={len(artist)} min={min(artist.values()) if artist else 0} max={max(artist.values()) if artist else 0}")
    print(f"unique_genre={len(genre)} min={min(genre.values()) if genre else 0} max={max(genre.values()) if genre else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
