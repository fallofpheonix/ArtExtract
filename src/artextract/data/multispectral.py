from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:  # pragma: no cover
    torch = None
    Dataset = object
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class MultiSpectralRecord:
    sample_id: str
    split: str
    channels: List[str]
    channel_paths: Dict[str, str]
    width: int
    height: int
    pigments: List[str]
    damage: bool
    restoration: bool
    hidden_image: bool
    hidden_gt_path: str | None


def _parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f", ""}:
        return False
    return default


def _parse_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def _parse_list(value: object) -> List[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            x = json.loads(s)
            if isinstance(x, list):
                return [str(v).strip() for v in x if str(v).strip()]
        except json.JSONDecodeError:
            pass
    sep = ";" if ";" in s else ","
    return [p.strip() for p in s.split(sep) if p.strip()]


def _parse_channels(value: object) -> List[str]:
    out = _parse_list(value)
    return [c.lower() for c in out]


def _extract_channel_paths(row: Dict[str, object]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in row.items():
        if not key.endswith("_path"):
            continue
        if key == "hidden_gt_path":
            continue
        name = key[: -len("_path")].strip().lower()
        s = "" if value is None else str(value).strip()
        if s:
            out[name] = s
    return out


def _record_from_dict(row: Dict[str, object]) -> MultiSpectralRecord:
    sample_id = str(row.get("sample_id", "")).strip()
    split = str(row.get("split", "train")).strip().lower() or "train"
    channels = _parse_channels(row.get("channels", ""))
    channel_paths = _extract_channel_paths(row)

    if not channels:
        channels = sorted(channel_paths.keys())
    if not sample_id:
        sample_id = f"sample_{abs(hash(json.dumps(row, sort_keys=True)))}"

    hidden_gt = row.get("hidden_gt_path")
    hidden_gt_path = str(hidden_gt).strip() if hidden_gt is not None else ""
    if not hidden_gt_path:
        hidden_gt_path = None

    return MultiSpectralRecord(
        sample_id=sample_id,
        split=split,
        channels=channels,
        channel_paths=channel_paths,
        width=_parse_int(row.get("width"), default=0),
        height=_parse_int(row.get("height"), default=0),
        pigments=_parse_list(row.get("pigments", "")),
        damage=_parse_bool(row.get("damage"), default=False),
        restoration=_parse_bool(row.get("restoration"), default=False),
        hidden_image=_parse_bool(row.get("hidden_image"), default=False),
        hidden_gt_path=hidden_gt_path,
    )


def load_multispectral_manifest(path: str | Path) -> List[MultiSpectralRecord]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")

    records: List[MultiSpectralRecord] = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("jsonl row must be an object")
                records.append(_record_from_dict(row))
    else:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(_record_from_dict(row))

    if not records:
        raise ValueError(f"manifest has no records: {p}")
    return records


def write_multispectral_manifest_csv(path: str | Path, records: Sequence[MultiSpectralRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    all_channels = sorted({c for r in records for c in r.channels})
    fieldnames = [
        "sample_id",
        "split",
        "channels",
        "width",
        "height",
        "pigments",
        "damage",
        "restoration",
        "hidden_image",
        "hidden_gt_path",
    ] + [f"{c}_path" for c in all_channels]

    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            row: Dict[str, object] = {
                "sample_id": r.sample_id,
                "split": r.split,
                "channels": json.dumps(r.channels),
                "width": r.width,
                "height": r.height,
                "pigments": json.dumps(r.pigments),
                "damage": int(r.damage),
                "restoration": int(r.restoration),
                "hidden_image": int(r.hidden_image),
                "hidden_gt_path": r.hidden_gt_path or "",
            }
            for c in all_channels:
                row[f"{c}_path"] = r.channel_paths.get(c, "")
            w.writerow(row)


def write_multispectral_manifest_jsonl(path: str | Path, records: Sequence[MultiSpectralRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            row = {
                "sample_id": r.sample_id,
                "split": r.split,
                "channels": r.channels,
                "width": r.width,
                "height": r.height,
                "pigments": r.pigments,
                "damage": r.damage,
                "restoration": r.restoration,
                "hidden_image": r.hidden_image,
                "hidden_gt_path": r.hidden_gt_path,
            }
            row.update({f"{k}_path": v for k, v in r.channel_paths.items()})
            f.write(json.dumps(row) + "\n")


def collect_pigment_vocab(records: Sequence[MultiSpectralRecord]) -> List[str]:
    vals = sorted({p for r in records for p in r.pigments})
    return vals


def split_records(records: Sequence[MultiSpectralRecord], split: str) -> List[MultiSpectralRecord]:
    s = split.strip().lower()
    return [r for r in records if r.split == s]


def _load_channel(path: Path, image_size: int) -> "torch.Tensor":
    try:
        from PIL import Image, ImageFile
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Pillow is required") from e

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    with Image.open(path) as img:
        img = img.convert("L")
        if image_size > 0:
            img = img.resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _resolve_path(base: Path, p: str) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    return (base / q).resolve()


class MultiSpectralDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        channels_order: Sequence[str],
        split: str,
        tasks: Iterable[str],
        pigments_vocab: Sequence[str] | None = None,
        image_size: int = 128,
        strict_dimensions: bool = False,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"torch unavailable: {_IMPORT_ERROR}")

        self.manifest_path = Path(manifest_path).resolve()
        self.channels_order = [c.lower() for c in channels_order]
        self.tasks = {t.strip().lower() for t in tasks}
        self.image_size = int(image_size)
        self.strict_dimensions = strict_dimensions

        records = load_multispectral_manifest(self.manifest_path)
        self.records = split_records(records, split)

        if "reconstruction" in self.tasks:
            self.records = [r for r in self.records if r.hidden_gt_path]

        if not self.records:
            raise ValueError(f"no records for split={split} tasks={sorted(self.tasks)}")

        if pigments_vocab is None:
            self.pigments_vocab = collect_pigment_vocab(self.records)
        else:
            self.pigments_vocab = list(pigments_vocab)
        self.pigment_index = {x: i for i, x in enumerate(self.pigments_vocab)}

    def __len__(self) -> int:
        return len(self.records)

    def _build_input(self, rec: MultiSpectralRecord) -> tuple["torch.Tensor", "torch.Tensor"]:
        base = self.manifest_path.parent
        planes: List[torch.Tensor] = []
        mask_vals: List[float] = []

        for ch in self.channels_order:
            path_s = rec.channel_paths.get(ch, "")
            has_path = bool(path_s)
            if has_path and ch not in rec.channels:
                # preserve deterministic contract: row-level channels field is authority
                has_path = False

            if has_path:
                p = _resolve_path(base, path_s)
                if p.exists() and p.suffix.lower() in _IMAGE_EXTS:
                    plane = _load_channel(p, self.image_size)
                    mask_vals.append(1.0)
                else:
                    plane = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
                    mask_vals.append(0.0)
            else:
                plane = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
                mask_vals.append(0.0)
            planes.append(plane)

        x = torch.stack(planes, dim=0)
        mask = torch.as_tensor(mask_vals, dtype=torch.float32)
        return x, mask

    def _build_targets(self, rec: MultiSpectralRecord) -> Dict[str, "torch.Tensor"]:
        out: Dict[str, torch.Tensor] = {}

        if "properties" in self.tasks:
            pigments = torch.zeros((len(self.pigments_vocab),), dtype=torch.float32)
            for p in rec.pigments:
                idx = self.pigment_index.get(p)
                if idx is not None:
                    pigments[idx] = 1.0
            out["pigments"] = pigments
            out["damage"] = torch.tensor(float(rec.damage), dtype=torch.float32)
            out["restoration"] = torch.tensor(float(rec.restoration), dtype=torch.float32)

        if "hidden" in self.tasks:
            out["hidden_image"] = torch.tensor(float(rec.hidden_image), dtype=torch.float32)

        if "reconstruction" in self.tasks:
            if not rec.hidden_gt_path:
                raise ValueError(f"reconstruction requires hidden_gt_path: {rec.sample_id}")
            p = _resolve_path(self.manifest_path.parent, rec.hidden_gt_path)
            gt_single = _load_channel(p, self.image_size)
            out["hidden_gt"] = gt_single.unsqueeze(0).repeat(len(self.channels_order), 1, 1)

        return out

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        if self.strict_dimensions and rec.width > 0 and rec.height > 0 and self.image_size > 0:
            if rec.width != self.image_size or rec.height != self.image_size:
                raise ValueError(
                    f"dimension mismatch sample_id={rec.sample_id} manifest={rec.width}x{rec.height} "
                    f"expected={self.image_size}x{self.image_size}"
                )

        x, channel_mask = self._build_input(rec)
        targets = self._build_targets(rec)

        return {
            "x": x,
            "channel_mask": channel_mask,
            "targets": targets,
            "meta": {
                "sample_id": rec.sample_id,
                "split": rec.split,
                "channels": rec.channels,
            },
        }


def multispectral_collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"torch unavailable: {_IMPORT_ERROR}")

    xs = torch.stack([b["x"] for b in batch], dim=0)
    masks = torch.stack([b["channel_mask"] for b in batch], dim=0)

    target_keys = sorted({k for b in batch for k in b["targets"].keys()})
    targets: Dict[str, torch.Tensor] = {}
    for k in target_keys:
        # target_keys was built from all samples, so at least one sample has key k
        ref = next((bb["targets"][k] for bb in batch if k in bb["targets"]), None)
        assert ref is not None, f"key '{k}' in target_keys but not found in any sample"
        vals = []
        for b in batch:
            t = b["targets"].get(k)
            if t is None:
                t = torch.zeros_like(ref)
            vals.append(t)
        targets[k] = torch.stack(vals, dim=0)

    meta = [b["meta"] for b in batch]
    return {
        "x": xs,
        "channel_mask": masks,
        "targets": targets,
        "meta": meta,
    }
