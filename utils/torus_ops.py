"""
Torus-linear map for seamless 45° digital camouflage.

torus_linear_map_inv_nn
────────────────────────
A tileable-preserving change of basis on a W×H torus.

Sampling matrix  [[1, -1], [1, 1]]  has INTEGER entries, which guarantees
that the W×H periodicity of the source image is exactly preserved in the output:
  • At xs + W :  u → (u+1) % 1 = u  → same source pixel ✓
  • At ys + H :  v → (v+1) % 1 = v  → same source pixel ✓

For each output pixel (xs, ys):
    u = xs/W,  v = ys/H
    x_src = (u − v) mod 1 · W
    y_src = (u + v) mod 1 · H

Effect: horizontal pixel bands in the source image appear as ≈45° diagonal
parallelogram bands in the output.

Used by DigitalCamoGenerator._diagonal_pixelate to produce the diagonal
digital camouflage from an axis-aligned base pattern.

Note: diagonal digitalization of arbitrary camos uses the direct diamond
algorithm in utils/image_ops.apply_diagonal_digitalization (no torus rotation
needed — it works entirely in the diagonal coordinate frame).
"""
from __future__ import annotations
import numpy as np


def torus_linear_map_inv_nn(img: np.ndarray) -> np.ndarray:
    """
    Tileable-preserving 45° diagonal view of a W×H-periodic image.

    Sampling matrix [[1,-1],[1,1]] — integer entries, exactly preserves torus.
    Uses floor (not rint) to avoid Voronoi-band discontinuities.
    Palette colours are preserved exactly (nearest-neighbour sampling).
    """
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]

    u = xs / W   # [0, 1)
    v = ys / H   # [0, 1)

    x = (u - v) % 1.0
    y = (u + v) % 1.0

    xi = np.floor(x * W).astype(np.int64) % W
    yi = np.floor(y * H).astype(np.int64) % H
    return img[yi, xi]
