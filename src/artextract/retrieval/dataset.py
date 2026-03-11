from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except Exception as e:  # pragma: no cover
    torch = None
    F = None
    Dataset = object
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class RetrievalSample:
    cover_path: str
    hidden_path: str


def _list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


class SyntheticHiddenRetrievalDataset(Dataset):
    """Synthetic hidden-image retrieval dataset.

    Each sample creates:
    - cover image C (visible painting)
    - hidden image H (underpainting)
    - random mask M and opacity a
    - observed O = C*(1-a*M) + H*(a*M)

    Model input: O
    Target: H
    """

    def __init__(
        self,
        images_root: str | Path,
        split: str,
        image_size: int = 128,
        train_split: float = 0.85,
        max_images: int = 0,
        seed: int = 42,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"torch unavailable: {_IMPORT_ERROR}")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.root = Path(images_root)
        self.image_size = int(image_size)
        self.seed = int(seed)

        files = _list_images(self.root)
        if max_images and max_images > 0:
            files = files[: int(max_images)]
        if len(files) < 2:
            raise ValueError(f"need at least 2 images, found {len(files)} under {self.root}")

        n_train = max(1, int(len(files) * float(train_split)))
        n_train = min(n_train, len(files) - 1)
        if split == "train":
            self.files = files[:n_train]
            self.pool = files[:n_train]
        else:
            self.files = files[n_train:]
            self.pool = files[n_train:]
            if len(self.files) < 2:
                # fallback so val exists even for tiny subsets
                self.files = files[-max(2, len(files) // 10) :]
                self.pool = self.files

    def __len__(self) -> int:
        return len(self.files)

    def _read_image(self, path: Path) -> "torch.Tensor":
        try:
            from PIL import Image, ImageFile
        except ImportError as e:
            raise RuntimeError("Pillow is required") from e

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        with Image.open(path) as img:
            img = img.convert("RGB").resize((self.image_size, self.image_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t

    def _random_mask(self, rng: np.random.Generator) -> "torch.Tensor":
        h = self.image_size
        w = self.image_size
        m = np.zeros((h, w), dtype=np.float32)

        n_shapes = int(rng.integers(2, 7))
        for _ in range(n_shapes):
            y0 = int(rng.integers(0, max(1, h - 16)))
            x0 = int(rng.integers(0, max(1, w - 16)))
            hh = int(rng.integers(max(8, h // 12), max(12, h // 3)))
            ww = int(rng.integers(max(8, w // 12), max(12, w // 3)))
            y1 = min(h, y0 + hh)
            x1 = min(w, x0 + ww)
            m[y0:y1, x0:x1] = 1.0

        mask = torch.from_numpy(m).unsqueeze(0)
        k = int(rng.integers(5, 15))
        if k % 2 == 0:
            k += 1
        mask = F.avg_pool2d(mask.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0)
        mask = mask.clamp(0.0, 1.0)
        return mask

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor", dict]:
        rng = np.random.default_rng(self.seed + idx)

        cover_path = self.files[idx]
        hidden_idx = int(rng.integers(0, len(self.pool)))
        hidden_path = self.pool[hidden_idx]
        if hidden_path == cover_path:
            hidden_path = self.pool[(hidden_idx + 1) % len(self.pool)]

        cover = self._read_image(cover_path)
        hidden = self._read_image(hidden_path)

        mask = self._random_mask(rng)
        alpha = float(rng.uniform(0.35, 0.75))

        observed = cover * (1.0 - alpha * mask) + hidden * (alpha * mask)
        noise_std = float(rng.uniform(0.0, 0.02))
        if noise_std > 0:
            observed = (observed + torch.randn_like(observed) * noise_std).clamp(0.0, 1.0)

        meta = {
            "cover_path": str(cover_path),
            "hidden_path": str(hidden_path),
        }
        return observed, hidden, meta
