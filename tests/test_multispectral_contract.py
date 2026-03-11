from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artextract.data import MultiSpectralDataset, load_multispectral_manifest
from artextract.data.multispectral import (
    MultiSpectralRecord,
    write_multispectral_manifest_csv,
)

try:
    import torch  # noqa: F401
except Exception:
    _HAS_TORCH = False
else:
    _HAS_TORCH = True


def _write_img(path: Path, value: int, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size, size), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


class MultiSpectralContractTests(unittest.TestCase):
    @unittest.skipUnless(_HAS_TORCH, "torch is required for dataset tensor tests")
    def test_manifest_parser_and_loader_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rgb = root / "rgb.png"
            ir = root / "ir.png"
            uv = root / "uv.png"
            xray = root / "xray.png"
            hidden = root / "hidden.png"

            _write_img(rgb, 10)
            _write_img(ir, 20)
            _write_img(uv, 30)
            _write_img(xray, 40)
            _write_img(hidden, 50)

            rec = MultiSpectralRecord(
                sample_id="s1",
                split="train",
                channels=["rgb", "ir", "uv", "xray"],
                channel_paths={
                    "rgb": str(rgb),
                    "ir": str(ir),
                    "uv": str(uv),
                    "xray": str(xray),
                },
                width=32,
                height=32,
                pigments=["lead_white"],
                damage=True,
                restoration=False,
                hidden_image=True,
                hidden_gt_path=str(hidden),
            )
            manifest = root / "manifest.csv"
            write_multispectral_manifest_csv(manifest, [rec])

            loaded = load_multispectral_manifest(manifest)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].channels, ["rgb", "ir", "uv", "xray"])

            ds = MultiSpectralDataset(
                manifest_path=manifest,
                channels_order=["rgb", "ir", "uv", "xray"],
                split="train",
                tasks=["properties", "hidden", "reconstruction"],
                image_size=32,
            )
            item = ds[0]
            self.assertEqual(tuple(item["x"].shape), (4, 32, 32))
            self.assertEqual(tuple(item["channel_mask"].shape), (4,))
            self.assertEqual(float(item["channel_mask"].sum().item()), 4.0)
            self.assertIn("hidden_gt", item["targets"])
            self.assertEqual(tuple(item["targets"]["hidden_gt"].shape), (4, 32, 32))

    @unittest.skipUnless(_HAS_TORCH, "torch is required for dataset tensor tests")
    def test_channel_mask_with_missing_modality(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rgb = root / "rgb.png"
            uv = root / "uv.png"
            xray = root / "xray.png"
            _write_img(rgb, 10)
            _write_img(uv, 30)
            _write_img(xray, 40)

            rec = MultiSpectralRecord(
                sample_id="s2",
                split="train",
                channels=["rgb", "uv", "xray"],
                channel_paths={
                    "rgb": str(rgb),
                    "uv": str(uv),
                    "xray": str(xray),
                },
                width=32,
                height=32,
                pigments=[],
                damage=False,
                restoration=False,
                hidden_image=False,
                hidden_gt_path=None,
            )
            manifest = root / "manifest.csv"
            write_multispectral_manifest_csv(manifest, [rec])

            ds = MultiSpectralDataset(
                manifest_path=manifest,
                channels_order=["rgb", "ir", "uv", "xray"],
                split="train",
                tasks=["properties", "hidden"],
                image_size=32,
            )
            item = ds[0]
            mask = item["channel_mask"].numpy().tolist()
            self.assertEqual(mask, [1.0, 0.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
