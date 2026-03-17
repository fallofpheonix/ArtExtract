from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from artextract.utils.fs import collect_image_paths


def _resolve_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    normalized = {name.lower(): name for name in columns}
    for alias in aliases:
        hit = normalized.get(alias.lower())
        if hit:
            return hit
    return None


def _object_id_from_filename(image_path: Path) -> int | None:
    match = re.match(r"^(\d+)", image_path.stem)
    return int(match.group(1)) if match else None


def _fallback_manifest(paths: list[Path]) -> pd.DataFrame:
    records = {
        "path": [str(path) for path in paths],
        "objectid": [_object_id_from_filename(path) for path in paths],
        "title": [path.name for path in paths],
        "artist": ["unknown"] * len(paths),
        "classification": ["unknown"] * len(paths),
        "style": ["unknown"] * len(paths),
    }
    return pd.DataFrame(records)


def build_manifest_table(images_dir: Path, opendata_dir: Path, max_images: int = 0) -> pd.DataFrame:
    image_paths = collect_image_paths(images_dir)
    if max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir.resolve()}")

    objects_csv = opendata_dir / "data" / "objects.csv"
    published_images_csv = opendata_dir / "data" / "published_images.csv"
    if not objects_csv.exists() or not published_images_csv.exists():
        return _fallback_manifest(image_paths)

    objects = pd.read_csv(objects_csv, low_memory=False)
    published = pd.read_csv(published_images_csv, low_memory=False)

    published_id_col = _resolve_column(published.columns, ["objectid", "depictstmsobjectid"])
    if published_id_col and published_id_col != "objectid":
        published = published.rename(columns={published_id_col: "objectid"})

    title_col = _resolve_column(objects.columns, ["title"]) or "title"
    artist_col = _resolve_column(objects.columns, ["attribution", "displayname", "artist"]) or "attribution"
    classification_col = _resolve_column(objects.columns, ["classification"]) or "classification"
    style_col = _resolve_column(objects.columns, ["style"]) or "style"

    selected_cols = ["objectid"]
    for column in (title_col, artist_col, classification_col, style_col):
        if column in objects.columns and column not in selected_cols:
            selected_cols.append(column)

    manifest = pd.DataFrame(
        {
            "path": [str(path) for path in image_paths],
            "objectid": [_object_id_from_filename(path) for path in image_paths],
        }
    )
    merged = manifest.merge(objects[selected_cols], on="objectid", how="left")

    renamed = merged.rename(
        columns={
            title_col: "title",
            artist_col: "artist",
            classification_col: "classification",
            style_col: "style",
        }
    )

    for column in ("title", "artist", "classification", "style"):
        if column not in renamed.columns:
            renamed[column] = "unknown"
        renamed[column] = renamed[column].fillna("unknown").astype(str)

    # TODO: validate joins against published_images.csv once NGA schema is stable.
    return renamed
