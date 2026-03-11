#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

HOSTS = ["uploads8", "uploads7", "uploads6", "uploads5"]


def urls_for_relpath(relpath: str) -> list[str]:
    fn = Path(relpath).name
    if "_" not in fn:
        return []
    artist, title = fn.split("_", 1)
    return [f"https://{h}.wikiart.org/images/{artist}/{title}" for h in HOSTS]


def load_relpaths(manifests: list[str]) -> list[str]:
    out = set()
    for mf in manifests:
        with open(mf, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                out.add(row["image_relpath"])
    return sorted(out)


def fetch_one(relpath: str, out_root: Path, timeout: float) -> tuple[str, str]:
    out = out_root / relpath
    if out.exists() and out.stat().st_size > 0:
        return "exists", relpath

    out.parent.mkdir(parents=True, exist_ok=True)
    for url in urls_for_relpath(relpath):
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(req, timeout=timeout) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if resp.status == 200 and ctype.startswith("image"):
                    data = resp.read()
                    if data:
                        out.write_bytes(data)
                        return "ok", relpath
        except URLError:
            continue
        except Exception:
            continue

    return "miss", relpath


def main() -> int:
    p = argparse.ArgumentParser(description="Download WikiArt images from manifest relpaths")
    p.add_argument("--manifest", action="append", required=True, help="Manifest CSV (repeatable)")
    p.add_argument("--out-root", default="data/raw/images")
    p.add_argument("--max-images", type=int, default=0, help="0 means all")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--timeout", type=float, default=15.0)
    args = p.parse_args()

    relpaths = load_relpaths(args.manifest)
    if args.max_images > 0:
        relpaths = relpaths[: args.max_images]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ok = 0
    exists = 0
    miss = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(fetch_one, rp, out_root, args.timeout) for rp in relpaths]
        for i, fut in enumerate(as_completed(futs), start=1):
            status, _ = fut.result()
            if status == "ok":
                ok += 1
            elif status == "exists":
                exists += 1
            else:
                miss += 1
            if i % 100 == 0 or i == len(relpaths):
                print(f"progress {i}/{len(relpaths)} ok={ok} exists={exists} miss={miss}")

    print(f"done total={len(relpaths)} ok={ok} exists={exists} miss={miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
