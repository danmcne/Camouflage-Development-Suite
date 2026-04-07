"""
Image operation helpers: blending, export, tiling, etc.
"""
from __future__ import annotations
import os
import cv2
import numpy as np
from PIL import Image


def export_png(img: np.ndarray, path: str, size: tuple[int, int] | None = None):
    """Save a BGR NumPy array as PNG, optionally resizing first."""
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(path, img)


def export_swatch_sheet(patterns: list[np.ndarray], path: str,
                         cols: int = 4, thumb_size: int = 256):
    """
    Create a contact sheet of pattern thumbnails and save as PNG.
    """
    from utils.rendering import make_thumbnail
    thumbs = [make_thumbnail(p, thumb_size) for p in patterns]
    rows   = (len(thumbs) + cols - 1) // cols
    sheet  = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * thumb_size:(r + 1) * thumb_size,
              c * thumb_size:(c + 1) * thumb_size] = t
    cv2.imwrite(path, sheet)


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    return img


def tile_pattern(pattern: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Tile pattern to fill target dimensions."""
    ph, pw = pattern.shape[:2]
    reps_y = (target_h + ph - 1) // ph
    reps_x = (target_w + pw - 1) // pw
    tiled  = np.tile(pattern, (reps_y, reps_x, 1))
    return tiled[:target_h, :target_w]
