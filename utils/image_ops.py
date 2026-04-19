"""
Image operation helpers: export, tiling, swatch sheet.

export_image() handles all formats and sizes:
  - PNG: preserves BGRA alpha channel if present
  - JPEG: composites alpha over white before saving (no alpha support)
  - TIFF: preserves BGRA alpha channel if present
"""
from __future__ import annotations
import os
import cv2
import numpy as np
from PIL import Image


EXPORT_SIZES = {
    "512 × 512":   (512,  512),
    "1024 × 1024": (1024, 1024),
    "2048 × 2048": (2048, 2048),
    "4096 × 4096": (4096, 4096),
    "8192 × 8192": (8192, 8192),
}


def _composite_over_white(img: np.ndarray) -> np.ndarray:
    """Flatten a BGRA image onto white. Returns BGR uint8."""
    if img.ndim == 3 and img.shape[2] == 4:
        a   = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        return (bgr * a + 255.0 * (1.0 - a)).clip(0, 255).astype(np.uint8)
    return img


def export_image(img: np.ndarray, path: str,
                 size: tuple[int, int] | None = None,
                 quality: int = 95) -> None:
    """
    Save img to path.  Format is inferred from extension.

    img may be BGR (H,W,3) or BGRA (H,W,4).

    PNG / TIFF  – alpha channel is preserved when present.
    JPEG        – alpha is composited over white (JPEG has no alpha).

    size: (width, height) to resize before saving; None = keep original.
    """
    if size is not None:
        interp = cv2.INTER_LANCZOS4
        if img.ndim == 3 and img.shape[2] == 4:
            # Resize BGRA maintaining alpha
            img = cv2.resize(img, size, interpolation=interp)
        else:
            img = cv2.resize(img, size, interpolation=interp)

    ext = os.path.splitext(path)[1].lower()

    if ext in (".jpg", ".jpeg"):
        # JPEG cannot store alpha — composite over white
        img_out = _composite_over_white(img)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        cv2.imwrite(path, img_out, encode_params)

    elif ext in (".tif", ".tiff"):
        if img.ndim == 3 and img.shape[2] == 4:
            # PIL handles TIFF with alpha better than OpenCV
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            Image.fromarray(rgba, "RGBA").save(path)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb, "RGB").save(path)

    else:  # PNG (default)
        cv2.imwrite(path, img)


# Keep old name as alias for backwards compatibility
def export_png(img: np.ndarray, path: str,
               size: tuple[int, int] | None = None) -> None:
    export_image(img, path, size)


def export_swatch_sheet(patterns: list[np.ndarray], path: str,
                         cols: int = 4, thumb_size: int = 256):
    from utils.rendering import make_thumbnail
    thumbs = [make_thumbnail(p if p.ndim == 3 and p.shape[2] == 3
                              else _composite_over_white(p), thumb_size)
              for p in patterns]
    rows  = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r*thumb_size:(r+1)*thumb_size, c*thumb_size:(c+1)*thumb_size] = t
    cv2.imwrite(path, sheet)


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    return img


def tile_pattern(pattern: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    ph, pw = pattern.shape[:2]
    reps_y = (target_h + ph - 1) // ph
    reps_x = (target_w + pw - 1) // pw
    tiled  = np.tile(pattern, (reps_y, reps_x, 1))
    return tiled[:target_h, :target_w]
