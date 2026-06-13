from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SplitRecord:
    image_relpath: str
    label_id: int


ClassMap = Dict[int, str]


def load_split_csv(path: str | Path) -> List[SplitRecord]:
    rows: List[SplitRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"invalid row in {path} at line {idx}: {row}")
            relpath = row[0].strip()
            label = int(row[1])
            rows.append(SplitRecord(image_relpath=relpath, label_id=label))
    return rows


def load_class_map(path: str | Path) -> ClassMap:
    out: ClassMap = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"invalid class map line {idx} in {path}: {line!r}")
            k = int(parts[0])
            out[k] = parts[1]
    return out
