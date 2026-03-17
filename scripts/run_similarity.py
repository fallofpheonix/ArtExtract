#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # Why: the repo still documents direct script execution from a checkout.
    sys.path.insert(0, str(SRC))


def main() -> int:
    try:
        from artextract.api.similarity_cli import main as run_retrieval_main
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
