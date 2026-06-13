from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    index: int
    score: float
    metadata: Dict[str, Any]


class BaseRetriever(abc.ABC):
    @abc.abstractmethod
    def embed_queries(self, queries: List[Any]) -> np.ndarray:
        """Embed queries into vectors."""
        pass

    @abc.abstractmethod
    def search(self, query: Any, k: int = 10) -> List[SearchResult]:
        """Search for top-k matches."""
        pass


class BaseIndex(abc.ABC):
    @abc.abstractmethod
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index."""
        pass

    @abc.abstractmethod
    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search vectors and return (distances, indices)."""
        pass
