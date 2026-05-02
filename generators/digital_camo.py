"""
Digital Camouflage Generator – pixelation-based hard-edge patterns.

Standard digital camo (MARPAT, CADPAT, etc.):
  1. Generate a seamlessly-tiling base noise field.
  2. Colour-quantise to the palette (threshold bands).
  3. Pixelate with nearest-neighbour down/up-scale.
  4. Snap each cell to the nearest palette colour.

Two angles: 0° (axis-aligned) and 45° (diagonal).

0° angle
────────
Nearest-divisor rounding ensures the cell size divides W and H exactly,
so the pixelated result is perfectly seamless.

45° angle (diagonal pixel grid)
──────────────────────────────
Uses the 3×3 tile approach for a perfectly seamless result:
  1. Generate base noise at W×H.
  2. Tile 3×3 → 3W × 3H.
  3. Rotate the tiled image 45° around its centre.
  4. Pixelate (cell_size × cell_size blocks) in the rotated frame.
  5. Crop centre W×H (no rotate-back → pixel grid stays diagonal).

This gives genuine diamond-shaped pixel blocks that tile seamlessly.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator, get_bg_params, apply_transparent_bg
from config.defaults import GENERATORS


def _nearest_divisor(value: int, target: int) -> int:
    """Return the divisor of `value` closest to `target`."""
    best, best_d = target, abs(value - target)
    for d in range(1, value + 1):
        if value % d == 0 and abs(d - target) < best_d:
            best, best_d = d, abs(d - target)
    return best


def _snap_to_palette(img_bgr: np.ndarray,
                     palette_bgr: np.ndarray) -> np.ndarray:
    flat  = img_bgr.reshape(-1, 3).astype(np.float32)
    pal_f = palette_bgr.astype(np.float32)
    diff  = flat[:, None, :] - pal_f[None, :, :]
    idx   = (diff ** 2).sum(axis=2).argmin(axis=1)
    return palette_bgr[idx].reshape(img_bgr.shape)


def _pixelate(img: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    H, W = img.shape[:2]
    small = cv2.resize(img, (max(1, W // cell_w), max(1, H // cell_h)),
                       interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def _base_noise(width, height, periods, seed) -> np.ndarray:
    try:
        import noise as noise_lib
        rng = np.random.default_rng(seed)
        bx  = float(rng.integers(0, periods * 100)) / 100.0
        by  = float(rng.integers(0, periods * 100)) / 100.0
        field = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            ny = (y / height) * periods
            for x in range(width):
                nx = (x / width) * periods
                field[y, x] = noise_lib.pnoise2(
                    nx + bx, ny + by, octaves=4, persistence=0.5,
                    lacunarity=2, repeatx=periods, repeaty=periods)
        mn, mx = field.min(), field.max()
        return (field - mn) / (mx - mn + 1e-8)
    except ImportError:
        rng    = np.random.default_rng(seed)
        raw    = rng.random((height, width)).astype(np.float32)
        k      = max(3, min(width, height) // 8 | 1)
        padded = np.pad(raw, k, mode="wrap")
        blurred = cv2.GaussianBlur(padded, (k, k), k / 6)
        field   = blurred[k:-k, k:-k]
        mn, mx  = field.min(), field.max()
        return (field - mn) / (mx - mn + 1e-8)


class DigitalCamoGenerator(BaseGenerator):
    name = "Digital Camo"
    description = (
        "Hard-edge pixelated digital camouflage (MARPAT/CADPAT style). "
        "Angle 0° = axis-aligned blocks, seamless. "
        "Angle 45° = diagonal pixel grid via 3×3 tile approach, also seamless."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["digital_camo"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        cell_size    = max(2, int(params.get("cell_size",    12)))
        noise_periods= max(1, int(params.get("noise_periods", 4)))
        num_levels   = max(1, int(params.get("num_levels",    2)))
        level_scale  = float(params.get("level_scale",        2.5))
        angle_str    = str(params.get("angle",                "0"))
        diagonal     = (angle_str.strip() == "45")
        transparent  = bool(params.get("transparent_bg",      False))
        seed         = int(params.get("seed",                  42))

        n   = max(1, len(colors))
        bg_idx, _ = get_bg_params(params, n)
        pal_bgr = np.array([(int(b), int(g), int(r))
                             for r, g, b in colors], dtype=np.uint8)

        # ── Base noise → colour-quantised image ───────────────────────────────
        field = _base_noise(width, height, noise_periods, seed)
        thresholds = np.linspace(0.0, 1.0 + 1e-6, n + 1)
        base_img   = np.zeros((height, width, 3), dtype=np.uint8)
        for i, (r, g, b) in enumerate(colors):
            mask = (field >= thresholds[i]) & (field < thresholds[i + 1])
            base_img[mask] = (int(b), int(g), int(r))

        if diagonal:
            result = self._diagonal_pixelate(base_img, pal_bgr, cell_size,
                                             width, height, num_levels, level_scale)
        else:
            result = self._axis_pixelate(base_img, pal_bgr, cell_size,
                                         width, height, num_levels, level_scale)

        if transparent:
            return apply_transparent_bg(result, colors, bg_idx)
        return result

    # ── 0° axis-aligned ───────────────────────────────────────────────────────

    def _axis_pixelate(self, img, pal_bgr, cell_size, W, H,
                       num_levels, level_scale):
        cw = max(1, _nearest_divisor(W, cell_size))
        ch = max(1, _nearest_divisor(H, cell_size))
        result = _snap_to_palette(_pixelate(img, cw, ch), pal_bgr)
        for level in range(1, num_levels):
            cs2 = max(1, int(cell_size * level_scale ** level))
            cw2 = max(1, _nearest_divisor(W, cs2))
            ch2 = max(1, _nearest_divisor(H, cs2))
            coarse = _snap_to_palette(_pixelate(img, cw2, ch2), pal_bgr)
            mask = np.any(coarse != result, axis=2)
            keep = np.random.default_rng(level).random(mask.shape) > (0.4 / level)
            mask &= ~keep
            result[mask] = coarse[mask]
        return result

    # ── 45° diagonal (produce axis-aligned, then 3×3 tile → rotate 45° → crop) ─

    def _diagonal_pixelate(self, img, pal_bgr, cell_size, W, H,
                           num_levels, level_scale):
        """
        Diagonal digital camo via torus_linear_map_inv_nn.

        torus_linear_map_inv_nn uses sampling matrix [[1,-1],[1,1]] — integer
        entries — so it is EXACTLY tileable-preserving: at xs+W and ys+H the
        source coordinate wraps by a whole integer, returning to the same pixel.

        Pipeline:
          1. Produce axis-aligned digital camo (seamless by exact-divisor construction).
          2. torus_linear_map_inv_nn(pix) — apply diagonal view transform.

        Effect: square pixel blocks → ≈45° parallelogram-shaped blocks.
        Result tiles with EXACTLY the same (W,H) period as the original camo.
        """
        from utils.torus_ops import torus_linear_map_inv_nn
        pix = self._axis_pixelate(img, pal_bgr, cell_size, W, H,
                                   num_levels, level_scale)
        return torus_linear_map_inv_nn(pix)
