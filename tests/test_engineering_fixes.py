"""Tests for the engineering fixes applied to the ArtExtract repository."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _write_img(path: Path, value: int = 128, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size, size), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


# ---------------------------------------------------------------------------
# Fix 1: UNetRetrieval is an alias for ReconstructionUNet
# ---------------------------------------------------------------------------
class TestUNetAlias(unittest.TestCase):
    @unittest.skipUnless(_HAS_TORCH, "torch is required")
    def test_unet_retrieval_is_reconstruction_unet(self) -> None:
        from artextract.retrieval.model import UNetRetrieval
        from artextract.reconstruction.unet import ReconstructionUNet
        # The alias must resolve to the canonical class
        self.assertIs(UNetRetrieval, ReconstructionUNet)

    @unittest.skipUnless(_HAS_TORCH, "torch is required")
    def test_unet_retrieval_forward(self) -> None:
        from artextract.retrieval import UNetRetrieval
        model = UNetRetrieval(in_channels=3, out_channels=3, base_channels=8)
        x = torch.rand(2, 3, 32, 32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 3, 32, 32))


# ---------------------------------------------------------------------------
# Fix 2: CRNN forward pass produces correct embedding dimension (2 * rnn_hidden)
# ---------------------------------------------------------------------------
class TestCRNNForwardPass(unittest.TestCase):
    @unittest.skipUnless(_HAS_TORCH, "torch is required")
    def test_crnn_embedding_dimension(self) -> None:
        from artextract.models.crnn import CRNNMultiTask
        rnn_hidden = 64
        model = CRNNMultiTask(
            style_classes=5,
            artist_classes=10,
            genre_classes=3,
            patch_grid=2,
            global_dim=128,
            patch_dim=64,
            rnn_hidden=rnn_hidden,
        )
        x = torch.rand(2, 3, 32, 32)
        out = model(x)
        # embedding dim should equal global_dim + 2 * rnn_hidden
        expected_dim = 128 + 2 * rnn_hidden
        self.assertEqual(out["embedding"].shape[-1], expected_dim)

    @unittest.skipUnless(_HAS_TORCH, "torch is required")
    def test_crnn_output_keys(self) -> None:
        from artextract.models.crnn import CRNNMultiTask
        model = CRNNMultiTask(style_classes=3, artist_classes=4, genre_classes=2)
        x = torch.rand(1, 3, 64, 64)
        out = model(x)
        for key in ("style", "artist", "genre", "embedding"):
            self.assertIn(key, out)


# ---------------------------------------------------------------------------
# Fix 3: multispectral_collate handles batches with different target keys
# ---------------------------------------------------------------------------
class TestMultispectralCollateRobust(unittest.TestCase):
    @unittest.skipUnless(_HAS_TORCH, "torch is required")
    def test_collate_missing_target_key_uses_zero(self) -> None:
        from artextract.data.multispectral import multispectral_collate

        t_pigments = torch.zeros(6, dtype=torch.float32)
        t_hidden = torch.tensor(1.0, dtype=torch.float32)
        t_damage = torch.tensor(0.0, dtype=torch.float32)
        t_restoration = torch.tensor(0.0, dtype=torch.float32)

        sample_a = {
            "x": torch.zeros(4, 8, 8),
            "channel_mask": torch.ones(4),
            "targets": {
                "pigments": t_pigments,
                "damage": t_damage,
                "restoration": t_restoration,
                "hidden_image": t_hidden,
            },
            "meta": {"sample_id": "a"},
        }
        # sample_b is missing "hidden_image" key
        sample_b = {
            "x": torch.zeros(4, 8, 8),
            "channel_mask": torch.ones(4),
            "targets": {
                "pigments": t_pigments.clone(),
                "damage": t_damage.clone(),
                "restoration": t_restoration.clone(),
            },
            "meta": {"sample_id": "b"},
        }

        batch = multispectral_collate([sample_a, sample_b])
        self.assertIn("hidden_image", batch["targets"])
        # The fallback value for the missing key must be zero
        self.assertEqual(float(batch["targets"]["hidden_image"][1].item()), 0.0)
        # The non-missing sample is unchanged
        self.assertEqual(float(batch["targets"]["hidden_image"][0].item()), 1.0)

    @unittest.skipUnless(_HAS_TORCH, "torch is required")
    def test_collate_consistent_shapes(self) -> None:
        from artextract.data.multispectral import multispectral_collate

        samples = [
            {
                "x": torch.rand(4, 8, 8),
                "channel_mask": torch.ones(4),
                "targets": {"damage": torch.tensor(float(i % 2))},
                "meta": {"sample_id": str(i)},
            }
            for i in range(3)
        ]
        batch = multispectral_collate(samples)
        self.assertEqual(tuple(batch["x"].shape), (3, 4, 8, 8))
        self.assertEqual(tuple(batch["targets"]["damage"].shape), (3,))


# ---------------------------------------------------------------------------
# Fix 4: retrieval dataset raises ValueError for invalid split
# ---------------------------------------------------------------------------
class TestRetrievalDatasetNoLeakage(unittest.TestCase):
    @unittest.skipUnless(_HAS_TORCH, "torch is required")
    def test_val_split_raises_on_too_few_images(self) -> None:
        from artextract.retrieval.dataset import SyntheticHiddenRetrievalDataset

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Create 3 images – with train_split=0.9, val gets only 0 or 1 image
            for i in range(3):
                arr = np.full((16, 16, 3), fill_value=i * 40, dtype=np.uint8)
                Image.fromarray(arr, mode="RGB").save(root / f"img_{i}.png")

            with self.assertRaises(ValueError) as ctx:
                SyntheticHiddenRetrievalDataset(
                    images_root=root,
                    split="val",
                    image_size=16,
                    train_split=0.9,
                )
            self.assertIn("Validation split", str(ctx.exception))


# ---------------------------------------------------------------------------
# Fix 7: classwise_isolation_outliers is exported from training package
# ---------------------------------------------------------------------------
class TestTrainingExports(unittest.TestCase):
    def test_classwise_isolation_outliers_exported(self) -> None:
        from artextract.training import classwise_isolation_outliers
        self.assertTrue(callable(classwise_isolation_outliers))

    def test_classwise_isolation_outliers_runs(self) -> None:
        from artextract.training import classwise_isolation_outliers
        from artextract.training.outliers import OutlierResult
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((50, 16)).astype(np.float32)
        y = np.array([i % 3 for i in range(50)], dtype=np.int64)
        results = classwise_isolation_outliers(emb, y, contamination=0.1, min_samples_per_class=5)
        self.assertIsInstance(results, list)
        # Each result should be an OutlierResult with index, class_id, score
        for r in results:
            self.assertIsInstance(r, OutlierResult)
            self.assertIsInstance(r.index, int)
            self.assertIsInstance(r.class_id, int)
            self.assertIsInstance(r.score, float)


if __name__ == "__main__":
    unittest.main()
