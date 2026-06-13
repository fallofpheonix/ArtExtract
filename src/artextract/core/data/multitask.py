from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .wikiart import SplitRecord, load_split_csv


@dataclass(frozen=True)
class MultiTaskRow:
    image_relpath: str
    style_label: int
    artist_label: int
    genre_label: int


def _index_by_path(records: Iterable[SplitRecord]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        out[r.image_relpath] = r.label_id
    return out


def build_multitask_rows(metadata_dir: str | Path, split: str) -> List[MultiTaskRow]:
    base = Path(metadata_dir)
    style = load_split_csv(base / f"style_{split}.csv")
    artist = _index_by_path(load_split_csv(base / f"artist_{split}.csv"))
    genre = _index_by_path(load_split_csv(base / f"genre_{split}.csv"))

    rows: List[MultiTaskRow] = []
    missing_artist = 0
    missing_genre = 0

    for rec in style:
        a = artist.get(rec.image_relpath)
        g = genre.get(rec.image_relpath)
        if a is None:
            missing_artist += 1
            continue
        if g is None:
            missing_genre += 1
            continue
        rows.append(
            MultiTaskRow(
                image_relpath=rec.image_relpath,
                style_label=rec.label_id,
                artist_label=a,
                genre_label=g,
            )
        )

    if missing_artist or missing_genre:
        print(
            f"warning: dropped rows missing labels: artist={missing_artist}, genre={missing_genre}"
        )

    return rows
