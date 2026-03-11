from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .multispectral import (
    MultiSpectralRecord,
    write_multispectral_manifest_csv,
    write_multispectral_manifest_jsonl,
)

try:
    from PIL import Image, ImageEnhance, ImageFilter
except Exception as e:  # pragma: no cover
    Image = None
    ImageEnhance = None
    ImageFilter = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SyntheticManifestPaths:
    csv_path: Path
    jsonl_path: Path


def _list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS])


def _to_gray_arr(img: "Image.Image") -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _save_plane(x: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (_clip01(x) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)


def _make_modalities(
    visible: "Image.Image",
    hidden: "Image.Image",
    hidden_present: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    v = _to_gray_arr(visible)
    h = _to_gray_arr(hidden)

    alpha = float(rng.uniform(0.35, 0.75)) if hidden_present else float(rng.uniform(0.0, 0.08))
    mix = _clip01((1.0 - alpha) * v + alpha * h)

    ir = _clip01(0.70 * mix + 0.30 * h)
    uv = _clip01(np.asarray(visible.filter(ImageFilter.FIND_EDGES).convert("L"), dtype=np.float32) / 255.0)
    xray = _clip01(0.45 * v + 0.55 * h + rng.normal(0.0, 0.03, size=v.shape).astype(np.float32))

    return {
        "rgb": mix,
        "ir": ir,
        "uv": uv,
        "xray": xray,
        "hidden_gt": h,
    }


def _pick_pigments(sample_id: str) -> list[str]:
    vocab = [
        "lead_white",
        "ultramarine",
        "vermilion",
        "ochre",
        "malachite",
        "lamp_black",
    ]
    idx = abs(hash(sample_id))
    return [vocab[idx % len(vocab)], vocab[(idx // len(vocab)) % len(vocab)]]


def generate_synthetic_multispectral_dataset(
    images_root: str | Path,
    out_root: str | Path,
    channels: Sequence[str],
    image_size: int = 128,
    max_samples: int = 200,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> SyntheticManifestPaths:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"Pillow unavailable: {_IMPORT_ERROR}")

    c_order = [c.strip().lower() for c in channels]
    if not c_order:
        raise ValueError("channels cannot be empty")

    src_root = Path(images_root)
    out_root = Path(out_root)
    files = _list_images(src_root)
    if len(files) < 4:
        raise ValueError(f"need at least 4 source images for synthetic set: found {len(files)}")

    if max_samples > 0:
        files = files[:max_samples]

    rng = np.random.default_rng(seed)
    records: List[MultiSpectralRecord] = []

    n_train = max(1, int(len(files) * train_ratio))
    n_train = min(n_train, len(files) - 1)

    for i, vis_path in enumerate(files):
        split = "train" if i < n_train else "val"
        hidden_idx = int(rng.integers(0, len(files)))
        if hidden_idx == i:
            hidden_idx = (hidden_idx + 1) % len(files)
        hid_path = files[hidden_idx]

        with Image.open(vis_path) as v_img:
            v_img = v_img.convert("RGB").resize((image_size, image_size))
            v_img = ImageEnhance.Contrast(v_img).enhance(1.05)
        with Image.open(hid_path) as h_img:
            h_img = h_img.convert("RGB").resize((image_size, image_size))

        hidden_present = bool(i % 2 == 0)
        maps = _make_modalities(v_img, h_img, hidden_present=hidden_present, rng=rng)

        sample_id = f"syn_{i:06d}"
        ch_paths: dict[str, str] = {}
        for ch in c_order:
            if ch not in {"rgb", "ir", "uv", "xray"}:
                continue
            p = out_root / "channels" / split / ch / f"{sample_id}_{ch}.png"
            _save_plane(maps[ch], p)
            ch_paths[ch] = str(p.resolve())

        hidden_gt_path: str | None
        if hidden_present:
            p_h = out_root / "hidden_gt" / split / f"{sample_id}_hidden_gt.png"
            _save_plane(maps["hidden_gt"], p_h)
            hidden_gt_path = str(p_h.resolve())
        else:
            hidden_gt_path = None

        records.append(
            MultiSpectralRecord(
                sample_id=sample_id,
                split=split,
                channels=c_order,
                channel_paths=ch_paths,
                width=image_size,
                height=image_size,
                pigments=_pick_pigments(sample_id),
                damage=bool(i % 5 == 0),
                restoration=bool(i % 7 == 0),
                hidden_image=hidden_present,
                hidden_gt_path=hidden_gt_path,
            )
        )

    manifest_dir = out_root / "manifests"
    csv_path = manifest_dir / "multispectral_synthetic.csv"
    jsonl_path = manifest_dir / "multispectral_synthetic.jsonl"
    write_multispectral_manifest_csv(csv_path, records)
    write_multispectral_manifest_jsonl(jsonl_path, records)

    return SyntheticManifestPaths(csv_path=csv_path, jsonl_path=jsonl_path)
