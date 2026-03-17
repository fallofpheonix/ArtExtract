from __future__ import annotations

from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path for path in images_dir.glob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
