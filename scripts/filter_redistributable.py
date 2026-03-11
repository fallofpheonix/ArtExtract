#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys

ALLOWED_STATUS = {"public_domain", "licensed", "permission_granted"}
ALLOWED_FLAG = {"true", "1", "yes", "y"}


def is_redistributable(row):
    status = (row.get("rights_status") or "").strip().lower()
    flag = (row.get("redistributable") or "").strip().lower()
    return status in ALLOWED_STATUS and flag in ALLOWED_FLAG


def main() -> int:
    p = argparse.ArgumentParser(description="Split provenance CSV by redistributability")
    p.add_argument("--input", required=True)
    p.add_argument("--out-allowed", required=True)
    p.add_argument("--out-blocked", required=True)
    args = p.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8", newline="") as f_in:
            reader = csv.DictReader(f_in)
            if not reader.fieldnames:
                print("error: missing header", file=sys.stderr)
                return 2

            with open(args.out_allowed, "w", encoding="utf-8", newline="") as f_ok, open(
                args.out_blocked, "w", encoding="utf-8", newline=""
            ) as f_no:
                w_ok = csv.DictWriter(f_ok, fieldnames=reader.fieldnames)
                w_no = csv.DictWriter(f_no, fieldnames=reader.fieldnames)
                w_ok.writeheader()
                w_no.writeheader()

                ok = 0
                no = 0
                for row in reader:
                    if is_redistributable(row):
                        w_ok.writerow(row)
                        ok += 1
                    else:
                        w_no.writerow(row)
                        no += 1

        print(f"done: redistributable={ok}, blocked={no}")
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
