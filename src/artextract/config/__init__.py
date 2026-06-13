"""Runtime configuration package."""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict

from .retrieval import RetrievalRuntimeConfig

@dataclass(frozen=True)
class Config:
    payload: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


def load_config(path: str | Path) -> Config:
    p = Path(path)
    suffix = p.suffix.lower()
    with p.open("r", encoding="utf-8") as f:
        if suffix == ".yaml":
            data = json.load(f)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "YAML config requires PyYAML. Install requirements or use .yaml config."
                ) from e
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"unsupported config format: {p}")
    if not isinstance(data, dict):
        raise ValueError(f"config must be mapping: {path}")
    return Config(payload=data)

__all__ = ["RetrievalRuntimeConfig", "Config", "load_config"]
