#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main() -> int:
    try:
        from artextract.similarity import run_similarity_main
        return int(run_similarity_main())
    except Exception as e:
        print(
            "error: similarity runtime dependencies are missing or failed to import.\n"
            "install optional deps: pip install -r requirements_similarity.txt\n"
            f"details: {e}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
