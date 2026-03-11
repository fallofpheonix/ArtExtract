#!/usr/bin/env python3
"""Download NGA image files referenced by published_images.csv.

Input:
  - <opendata-dir>/data/published_images.csv
Output:
  - local image directory (default: ./images)

Design:
  - deterministic row ordering
  - id-based filenames for stable joins
  - concurrent downloads with retries
  - resumable (skips existing files)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download NGA painting images from published_images.csv")
    p.add_argument("--opendata-dir", default="nga_data", help="Path to cloned NGA opendata repo")
    p.add_argument(
        "--published-csv",
        default="",
        help="Explicit path to published_images.csv (overrides --opendata-dir)",
    )
    p.add_argument(
        "--no-auto-fetch-csv",
        action="store_false",
        dest="auto_fetch_csv",
        help="Disable fallback fetch from GitHub raw URL when local CSV is missing",
    )
    p.add_argument(
        "--published-csv-url",
        default="https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/published_images.csv",
        help="Fallback URL for published_images.csv when local file is unavailable",
    )
    p.add_argument("--output-dir", default="images", help="Output image directory")
    p.add_argument("--max-images", type=int, default=0, help="Max images to download (0 = all)")
    p.add_argument("--workers", type=int, default=16, help="Parallel download workers")
    p.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    p.add_argument("--retries", type=int, default=3, help="Retry count per file")
    p.add_argument("--iiif-size", default="!512,512", help="IIIF size token to inject into /full/<size>/0/")
    return p.parse_args()


def detect_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def maybe_rewrite_iiif(url: str, iiif_size: str) -> str:
    marker = "/full/"
    if marker not in url:
        return url
    # Rewrite first /full/<token>/0/ segment.
    head, tail = url.split(marker, 1)
    tail_parts = tail.split("/", 2)
    if len(tail_parts) < 3:
        return url
    return f"{head}{marker}{iiif_size}/0/{tail_parts[2]}"


def load_rows(csv_path: Path, max_images: int) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        oid_col = detect_col(cols, ["objectid", "depictstmsobjectid", "depictsTmsObjectID"])
        url_col = detect_col(cols, ["iiifthumburl", "iiifurl", "image_url", "url", "download_url"])
        if oid_col is None or url_col is None:
            raise RuntimeError(f"Required columns not found. Columns={cols}")

        dedup: Dict[str, Dict[str, str]] = {}
        for row in reader:
            oid = (row.get(oid_col) or "").strip()
            url = (row.get(url_col) or "").strip()
            if not oid or not url:
                continue
            if oid not in dedup:
                dedup[oid] = {"objectid": oid, "url": url}

        rows = [dedup[k] for k in sorted(dedup.keys(), key=lambda x: int(x) if x.isdigit() else x)]
        if max_images > 0:
            rows = rows[:max_images]
        return rows


def ensure_csv(csv_path: Path, auto_fetch: bool, csv_url: str, cache_dir: Path) -> Path:
    if csv_path.exists():
        return csv_path
    if not auto_fetch:
        raise FileNotFoundError(str(csv_path))

    cache_dir.mkdir(parents=True, exist_ok=True)
    fallback = cache_dir / "published_images.csv"
    print(f"Local CSV missing: {csv_path}")
    print(f"Fetching fallback CSV: {csv_url}")
    req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError("Fallback CSV download returned empty body")
    fallback.write_bytes(data)
    print(f"Saved fallback CSV: {fallback}")
    return fallback


def fetch_one(row: Dict[str, str], output_dir: Path, timeout: int, retries: int, iiif_size: str) -> Tuple[str, str]:
    oid = row["objectid"]
    url = maybe_rewrite_iiif(row["url"], iiif_size)

    ext = Path(url.split("?", 1)[0]).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
        ext = ".jpg"

    dst = output_dir / f"{oid}{ext}"
    if dst.exists() and dst.stat().st_size > 0:
        return ("skip", oid)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_err = ""

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError("empty response")
            tmp = dst.with_suffix(dst.suffix + ".part")
            with tmp.open("wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return ("ok", oid)
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2.0 * attempt, 6.0))

    return ("fail", f"{oid}\t{url}\t{last_err}")


def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"

    if args.published_csv:
        csv_path = Path(args.published_csv)
    else:
        csv_path = Path(args.opendata_dir) / "data" / "published_images.csv"

    try:
        csv_path = ensure_csv(
            csv_path=csv_path,
            auto_fetch=args.auto_fetch_csv,
            csv_url=args.published_csv_url,
            cache_dir=cache_dir,
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not resolve published_images.csv ({type(e).__name__}: {e})", file=sys.stderr)
        return 2

    rows = load_rows(csv_path, args.max_images)
    total = len(rows)
    if total == 0:
        print("No rows found in published_images.csv", file=sys.stderr)
        return 3

    print(f"Rows selected: {total:,}")
    print(f"Output dir   : {out_dir.resolve()}")
    print(f"Workers      : {args.workers}")

    ok = 0
    skip = 0
    fail = 0
    failures: List[str] = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(fetch_one, row, out_dir, args.timeout, args.retries, args.iiif_size)
            for row in rows
        ]
        for i, fut in enumerate(as_completed(futs), start=1):
            status, payload = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                failures.append(payload)

            if i % 200 == 0 or i == total:
                print(f"[{i:>6}/{total}] ok={ok} skip={skip} fail={fail}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s: ok={ok}, skip={skip}, fail={fail}")

    if failures:
        log_path = out_dir / "download_failures.tsv"
        log_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"Failure log: {log_path}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
