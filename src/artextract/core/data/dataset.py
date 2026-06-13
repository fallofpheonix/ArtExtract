from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:  # pragma: no cover
    torch = None
    Dataset = object
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class MultiTaskSample:
    image_relpath: str
    style_label: int
    artist_label: int
    genre_label: int


def read_manifest(manifest_csv: str | Path) -> List[MultiTaskSample]:
    rows: List[MultiTaskSample] = []
    with Path(manifest_csv).open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"image_relpath", "style_label", "artist_label", "genre_label"}
        if not required.issubset(r.fieldnames or set()):
            raise ValueError(
                f"manifest missing required columns: {required}, got {r.fieldnames}"
            )
        for row in r:
            rows.append(
                MultiTaskSample(
                    image_relpath=row["image_relpath"],
                    style_label=int(row["style_label"]),
                    artist_label=int(row["artist_label"]),
                    genre_label=int(row["genre_label"]),
                )
            )
    return rows


class MultiTaskImageDataset(Dataset):
    """Image dataset for multi-task style/artist/genre training."""

    def __init__(
        self,
        manifest_csv: str | Path,
        images_root: str | Path,
        transform=None,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"torch unavailable: {_IMPORT_ERROR}")

        self.samples = read_manifest(manifest_csv)
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        try:
            from PIL import Image, ImageFile
        except ImportError as e:
            raise RuntimeError("Pillow is required for image loading") from e

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        n = len(self.samples)
        if n == 0:
            raise IndexError("empty dataset")

        s = self.samples[idx]
        path = self.images_root / s.image_relpath
        if not path.exists():
            raise FileNotFoundError(f"Missing image at {path}")
            
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Corrupted image at {path}: {e}")

        if self.transform is not None:
            img = self.transform(img)

        labels = {
            "style": torch.tensor(s.style_label, dtype=torch.long),
            "artist": torch.tensor(s.artist_label, dtype=torch.long),
            "genre": torch.tensor(s.genre_label, dtype=torch.long),
        }
        return img, labels
