from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

from .base import BaseRetriever


class CLIPRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        backend: str = "open_clip",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ):
        self.device = device
        self.model, self.preprocess, self.backend_name = self._load_model(
            model_name, backend, pretrained, device
        )

    def _load_model(self, model_name, backend, pretrained, device):
        if backend == "openai":
            import clip
            model, preprocess = clip.load(model_name, device=device)
            return model, preprocess, f"openai:{model_name}"
        else:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name.replace("/", "-"),
                pretrained=pretrained,
                device=device,
            )
            model.eval()
            return model, preprocess, f"open_clip:{model_name}:{pretrained}"

    @torch.inference_mode()
    def embed_images(self, paths: List[str | Path], batch_size: int = 64) -> Tuple[np.ndarray, List[int]]:
        feats = []
        kept = []
        for start in tqdm(range(0, len(paths), batch_size), desc="Embedding"):
            cur = paths[start : start + batch_size]
            batch = []
            loc = []
            for j, p in enumerate(cur):
                try:
                    img = Image.open(p).convert("RGB")
                    batch.append(self.preprocess(img))
                    loc.append(start + j)
                except (FileNotFoundError, OSError, UnidentifiedImageError):
                    continue
            if not batch:
                continue
            x = torch.stack(batch).to(self.device)
            y = self.model.encode_image(x)
            y = y / y.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            feats.append(y.cpu().numpy().astype("float32"))
            kept.extend(loc)
        
        if not feats:
            raise RuntimeError("No embeddings generated")
        emb = np.vstack(feats)
        return emb, kept

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        # Placeholder for text embedding if needed
        raise NotImplementedError("Text embedding not yet implemented in CLIPRetriever")

    def search(self, query: Any, k: int = 10) -> List[Any]:
        raise NotImplementedError("Use CLIPRetriever with an Index for searching")
