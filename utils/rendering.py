"""
Rendering utilities: NumPy ↔ Qt conversions, superimpose, thumbnail helpers.
"""
from __future__ import annotations
import numpy as np
import cv2
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


def bgr_to_qpixmap(img: np.ndarray, max_size: int | None = None) -> QPixmap:
    """Convert a uint8 BGR NumPy image to QPixmap."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    if max_size is not None:
        pixmap = pixmap.scaled(
            max_size, max_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pixmap


def qpixmap_to_bgr(pixmap: QPixmap) -> np.ndarray:
    """Convert a QPixmap back to uint8 BGR NumPy array."""
    qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 3)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def superimpose(
    pattern: np.ndarray,
    background: np.ndarray,
    alpha: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    """
    Overlay a pattern on a background.

    Args:
        pattern:    uint8 BGR pattern image
        background: uint8 BGR background image
        alpha:      opacity of pattern (0=invisible, 1=fully opaque)
        offset_x/y: pixel offset (useful for tiling experiments)

    Returns:
        uint8 BGR composite image (same size as background)
    """
    bg = background.copy()
    bh, bw = bg.shape[:2]

    # Resize pattern to background size if needed
    pat = cv2.resize(pattern, (bw, bh), interpolation=cv2.INTER_LINEAR)

    # Apply offset via roll
    if offset_x or offset_y:
        pat = np.roll(pat, offset_x, axis=1)
        pat = np.roll(pat, offset_y, axis=0)

    # Alpha blend
    if alpha >= 1.0:
        return pat
    return cv2.addWeighted(bg, 1.0 - alpha, pat, alpha, 0)


def make_thumbnail(img: np.ndarray, size: int = 128) -> np.ndarray:
    """Return a square thumbnail (centre-cropped then resized)."""
    h, w = img.shape[:2]
    # Centre crop to square
    m = min(h, w)
    y0, x0 = (h - m) // 2, (w - m) // 2
    cropped = img[y0:y0 + m, x0:x0 + m]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
