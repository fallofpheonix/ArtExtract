#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# ROOT = Path(__file__).resolve().parents[1] # Removed path hack
# SRC = ROOT / "src"


def main() -> int:
    try:
        from artextract.retrieval.cli import main as run_retrieval_main
        return int(run_retrieval_main())
    except Exception as e:
        print(
            "error: retrieval runtime dependencies are missing or failed to import.\n"
            f"details: {e}",
            file=sys.stderr,
        )
        return 2



if __name__ == "__main__":
    raise SystemExit(main())
