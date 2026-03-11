from __future__ import annotations

import unittest
import sys
from pathlib import Path

try:
    import torch
except Exception:
    _HAS_TORCH = False
else:
    _HAS_TORCH = True

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artextract.models import MultiSpectralMultiTaskModel


class MultiSpectralModelTests(unittest.TestCase):
    @unittest.skipUnless(_HAS_TORCH, "torch is required")
    def test_forward_shapes_and_gradients(self) -> None:
        model = MultiSpectralMultiTaskModel(
            in_channels=4,
            num_pigments=6,
            enable_properties=True,
            enable_hidden=True,
            enable_reconstruction=True,
            base_channels=8,
        )

        x = torch.rand(3, 4, 32, 32)
        mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        out = model(x, mask)
        self.assertEqual(tuple(out["pigments_logits"].shape), (3, 6))
        self.assertEqual(tuple(out["damage_logits"].shape), (3,))
        self.assertEqual(tuple(out["restoration_logits"].shape), (3,))
        self.assertEqual(tuple(out["hidden_logits"].shape), (3,))
        self.assertEqual(tuple(out["reconstruction"].shape), (3, 4, 32, 32))

        loss = (
            out["pigments_logits"].mean()
            + out["damage_logits"].mean()
            + out["restoration_logits"].mean()
            + out["hidden_logits"].mean()
            + out["reconstruction"].mean()
        )
        loss.backward()

        nonzero_grads = 0
        for p in model.parameters():
            if p.grad is not None and torch.any(p.grad != 0):
                nonzero_grads += 1
        self.assertGreater(nonzero_grads, 0)


if __name__ == "__main__":
    unittest.main()
