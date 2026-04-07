"""
BackgroundManager – loads, caches, and serves background images for fitness evaluation.
"""
from __future__ import annotations
import os
import cv2
import numpy as np


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class BackgroundManager:
    def __init__(self):
        self._paths: list[str] = []
        self._cache: dict[str, np.ndarray] = {}   # path → resized BGR image
        self._active_index: int = 0

    # ── loading ───────────────────────────────────────────────────────────────

    def add_folder(self, folder: str):
        """Scan a folder and add all supported image files."""
        for fname in sorted(os.listdir(folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                full = os.path.join(folder, fname)
                if full not in self._paths:
                    self._paths.append(full)

    def add_file(self, path: str):
        if path not in self._paths:
            self._paths.append(path)

    def remove(self, index: int):
        if 0 <= index < len(self._paths):
            path = self._paths.pop(index)
            self._cache.pop(path, None)
            self._active_index = min(self._active_index, max(0, len(self._paths) - 1))

    def clear(self):
        self._paths.clear()
        self._cache.clear()
        self._active_index = 0

    # ── retrieval ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._paths)

    def get_image(self, index: int, size: tuple[int, int] = (512, 512)) -> np.ndarray | None:
        """Return a resized BGR image for the given index (cached)."""
        if not self._paths or index >= len(self._paths):
            return None
        path = self._paths[index]
        cache_key = f"{path}@{size[0]}x{size[1]}"
        if cache_key not in self._cache:
            img = cv2.imread(path)
            if img is None:
                return None
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            self._cache[cache_key] = img
        return self._cache[cache_key]

    def get_active(self, size: tuple[int, int] = (512, 512)) -> np.ndarray | None:
        return self.get_image(self._active_index, size)

    def get_thumbnail(self, index: int, thumb: int = 96) -> np.ndarray | None:
        return self.get_image(index, (thumb, thumb))

    def set_active(self, index: int):
        self._active_index = max(0, min(index, len(self._paths) - 1))

    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def paths(self) -> list[str]:
        return list(self._paths)

    def random_image(self, size: tuple[int, int] = (512, 512)) -> np.ndarray | None:
        import random
        if not self._paths:
            return None
        return self.get_image(random.randrange(len(self._paths)), size)
