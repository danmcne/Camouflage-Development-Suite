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


def apply_digitalization(img: np.ndarray, level: int) -> np.ndarray:
    """
    Axis-aligned pixelation: subsample every 2^level pixels then upscale NN.
    Palette colours are preserved exactly. level 0 = no-op.
    """
    if level <= 0:
        return img
    step = 1 << level
    H, W = img.shape[:2]
    small = img[::step, ::step]
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def apply_diagonal_digitalization(img: np.ndarray, level: int) -> np.ndarray:
    """
    Diagonal (45°) block-quantisation using integer-lattice diamond tiles.

    Root cause of the "X seam": using fractional normalised coordinates
    (xs/W ± ys/H) introduces branch cuts at xs/W = ys/H and xs/W = 1-ys/H
    — two diagonal lines that cross at the image centre, forming an X.
    The underlying transform has determinant 2 (not ±1), so it identifies
    pairs of points across those lines, creating visible discontinuities.

    Fix: stay entirely in Z² — use integer diagonal coordinates with
    no fractional normalisation.  The integer lattice has no branch cuts.

    Algorithm:
      d1 = (xs + ys) mod 2W      ← one integer diagonal coordinate
      d2 = (xs − ys) mod 2H      ← other integer diagonal coordinate
      Quantise: q1 = (d1 // step) * step
                q2 = (d2 // step) * step
      Invert (exact, step is even so division by 2 is exact):
                xi = ((q1 + q2) // 2) % W
                yi = ((q1 − q2) // 2) % H
      Output pixel → img[yi, xi]

    Tileability: at xs+W, d1 shifts by W → q1 shifts by W (since step|W
    for power-of-2 step and power-of-2 canvas) → xi unchanged mod W ✓.
    Same at ys+H. Output is EXACTLY W×H-periodic with no X seam.

    Palette colours are preserved exactly (nearest-neighbour, no blending).
    """
    if level <= 0:
        return img.copy()
    step = 1 << level   # always even for level >= 1
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]

    # Integer diagonal coordinates — no fractional normalisation
    d1 = (xs + ys) % (2 * W)
    d2 = (xs - ys) % (2 * H)

    # Quantise to diamond block (q1, q2 are both multiples of step)
    q1 = (d1 // step) * step
    q2 = (d2 // step) * step

    # Invert: d1 = xs+ys, d2 = xs-ys  →  xs = (d1+d2)/2, ys = (d1-d2)/2
    # step is even → q1, q2 are even → division by 2 is always exact
    xi = ((q1 + q2) // 2) % W
    yi = ((q1 - q2) // 2) % H

    return img[yi.astype(np.int64), xi.astype(np.int64)]


def tile_pattern(pattern: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    ph, pw = pattern.shape[:2]
    reps_y = (target_h + ph - 1) // ph
    reps_x = (target_w + pw - 1) // pw
    tiled  = np.tile(pattern, (reps_y, reps_x, 1))
    return tiled[:target_h, :target_w]
