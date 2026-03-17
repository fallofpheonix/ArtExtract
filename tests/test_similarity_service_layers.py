from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
from typing import Any

import numpy as np
from PIL import Image

try:
    from artextract.config import RetrievalRuntimeConfig
    from artextract.core.retrieval_manifest import build_manifest_table
    from artextract.services.similarity_service import SimilarityRetrievalService
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from artextract.config import RetrievalRuntimeConfig
    from artextract.core.retrieval_manifest import build_manifest_table
    from artextract.services.similarity_service import SimilarityRetrievalService


class _FakeRetriever:
    def __init__(self, device: str):
        self.device = device

    def embed_images(self, *args: Any, **kwargs: Any):
        return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32), [0, 1]


class TestRetrievalManifest(unittest.TestCase):
    def test_manifest_falls_back_when_metadata_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images_dir = root / "images"
            images_dir.mkdir()
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(images_dir / "100_demo.png")

            manifest = build_manifest_table(images_dir=images_dir, opendata_dir=root, max_images=0)

            self.assertEqual(len(manifest), 1)
            self.assertIn("title", manifest.columns)
            self.assertEqual(manifest.loc[0, "artist"], "unknown")


class TestSimilarityService(unittest.TestCase):
    def test_service_handles_missing_label_column(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RetrievalRuntimeConfig(
                images_dir=root / "images",
                opendata_dir=root / "metadata",
                out_dir=root / "out",
                batch_size=2,
                max_images=0,
                top_k=3,
                label_col="non_existent_label",
                clusters=1,
            )

            with patch("artextract.services.similarity_service.build_manifest_table") as build_manifest:
                import pandas as pd

                build_manifest.return_value = pd.DataFrame(
                    {"path": ["a.png", "b.png"], "objectid": [1, 2], "classification": ["x", "y"]}
                )

                service = SimilarityRetrievalService(config, retriever_factory=_FakeRetriever)
                result = service.run()

                self.assertEqual(result["indexed"], 2)
                self.assertIn(result["index_backend"], {"faiss", "none"})
                self.assertTrue((config.out_dir / "embedding_metadata.csv").exists())


if __name__ == "__main__":
    unittest.main()
