from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class MultiSpectralCliSmokeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is required")
    def test_train_and_eval_cli_smoke(self) -> None:
        repo = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images_root = root / "images"
            images_root.mkdir(parents=True, exist_ok=True)

            for i in range(20):
                arr = np.full((40, 40, 3), fill_value=i * 5 % 255, dtype=np.uint8)
                Image.fromarray(arr, mode="RGB").save(images_root / f"img_{i:04d}.png")

            cfg = {
                "seed": 42,
                "model": {"image_size": 32, "base_channels": 8, "strict_dimensions": False},
                "training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "lr": 0.001,
                    "weight_decay": 0.0,
                    "num_workers": 0,
                    "train_ratio": 0.8,
                    "device": "cpu",
                },
                "loss_weights": {"property": 1.0, "hidden": 1.0, "reconstruction": 1.0},
            }
            cfg_path = root / "cfg.json"
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

            out_dir = root / "run"

            train_cmd = [
                sys.executable,
                str(repo / "scripts" / "train.py"),
                "--config",
                str(cfg_path),
                "--channels",
                "rgb,ir,uv,xray",
                "--tasks",
                "properties,hidden,reconstruction",
                "--synthetic-images-root",
                str(images_root),
                "--synthetic-out-root",
                str(root / "synthetic"),
                "--synthetic-max-samples",
                "16",
                "--out-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            subprocess.run(train_cmd, check=True, cwd=repo)

            eval_cmd = [
                sys.executable,
                str(repo / "scripts" / "eval.py"),
                "--manifest",
                str(out_dir / "resolved_manifest.csv"),
                "--checkpoint",
                str(out_dir / "best_model.pt"),
                "--pigments-vocab",
                str(out_dir / "pigments_vocab.json"),
                "--channels",
                "rgb,ir,uv,xray",
                "--tasks",
                "properties,hidden,reconstruction",
                "--config",
                str(cfg_path),
                "--out-dir",
                str(out_dir),
                "--device",
                "cpu",
            ]
            subprocess.run(eval_cmd, check=True, cwd=repo)

            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "run_meta.json").exists())
            self.assertTrue((out_dir / "confusion_matrix.png").exists())
            self.assertTrue((out_dir / "roc_curve.png").exists())
            self.assertTrue((out_dir / "recon_examples").exists())


if __name__ == "__main__":
    unittest.main()
