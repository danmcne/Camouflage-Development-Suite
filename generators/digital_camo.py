"""
Digital Camouflage Generator – pixelation-based hard-edge patterns.

Standard digital camo (MARPAT, CADPAT, etc.) is produced by:
  1. Generating a base noise/colour field at full size
  2. Downscaling to a small "cell" grid with nearest-neighbour (no blending)
  3. Snapping each cell to the nearest palette colour
  4. Upscaling back to full size with nearest-neighbour (no interpolation)

Toroidal guarantee: the base noise is generated with integer-period pnoise2;
nearest-neighbour resize of a tileable image is also tileable as long as
the cell size divides the canvas exactly.  We therefore round cell_size to the
nearest divisor of width/height.

Diagonal digital camo:
  Rotate the canvas by the chosen angle → pixelate → rotate back.
  The rotation+crop introduces a small seam; we mitigate it by padding the
  rotated canvas before pixelating and cropping after rotation-back.
  Not perfectly seamless, but very close for moderate angles.

Pixel sampling vs palette snap:
  After downscaling we snap each pixel to the nearest palette colour in RGB
  distance.  This keeps the output strictly on-palette regardless of the
  base noise's intermediate values.
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
    """
    Replace every pixel with the nearest palette colour (L2 in BGR space).
    img_bgr : (H, W, 3) uint8
    palette_bgr: (K, 3) uint8
    """
    flat  = img_bgr.reshape(-1, 3).astype(np.float32)
    pal_f = palette_bgr.astype(np.float32)
    diff  = flat[:, None, :] - pal_f[None, :, :]   # (N, K, 3)
    idx   = (diff**2).sum(axis=2).argmin(axis=1)    # (N,)
    return palette_bgr[idx].reshape(img_bgr.shape)


def _pixelate(img: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    """Nearest-neighbour downscale then upscale — the core digital-camo step."""
    H, W = img.shape[:2]
    small = cv2.resize(img,   (W // cell_w, H // cell_h),
                       interpolation=cv2.INTER_NEAREST)
    big   = cv2.resize(small, (W, H),
                       interpolation=cv2.INTER_NEAREST)
    return big


def _base_noise(width, height, periods, seed) -> np.ndarray:
    """
    Fast seamlessly-tiling base field using integer-period pnoise2.
    Falls back to blurred random noise if the `noise` library is absent.
    Returns float32 field in [0, 1].
    """
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
        blurred= cv2.GaussianBlur(padded, (k, k), k / 6)
        field  = blurred[k:-k, k:-k]
        mn, mx = field.min(), field.max()
        return (field - mn) / (mx - mn + 1e-8)


class DigitalCamoGenerator(BaseGenerator):
    name = "Digital Camo"
    description = (
        "Hard-edge pixelated digital camouflage. Produces MARPAT/CADPAT-style "
        "blocky patterns. Diagonal mode rotates the pixel grid for angular effect."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["digital_camo"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        cell_size    = int(params.get("cell_size",    12))
        noise_periods= int(params.get("noise_periods", 4))
        num_levels   = int(params.get("num_levels",    2))
        level_scale  = float(params.get("level_scale", 2.5))
        angle        = float(params.get("angle",       0.0))
        transparent  = bool(params.get("transparent_bg", False))
        seed         = int(params.get("seed",          42))

        rng = np.random.default_rng(seed)
        n   = max(1, len(colors))
        bg_idx, _ = get_bg_params(params, n)

        # Build BGR palette
        pal_bgr = np.array([(int(b), int(g), int(r))
                             for r, g, b in colors], dtype=np.uint8)

        # ── Base noise field → colourise by threshold ─────────────────────────
        periods = max(1, noise_periods)
        field   = _base_noise(width, height, periods, seed)
        thresholds = np.linspace(0.0, 1.0 + 1e-6, n + 1)
        base_img   = np.zeros((height, width, 3), dtype=np.uint8)
        for i, (r, g, b) in enumerate(colors):
            mask = (field >= thresholds[i]) & (field < thresholds[i+1])
            base_img[mask] = (int(b), int(g), int(r))

        # ── Pixelate (possibly at an angle) ───────────────────────────────────
        if abs(angle) > 1.0:
            result = self._diagonal_pixelate(
                base_img, pal_bgr, cell_size, angle, num_levels, level_scale, rng)
        else:
            result = self._axis_pixelate(
                base_img, pal_bgr, cell_size, width, height,
                num_levels, level_scale)

        if transparent:
            return apply_transparent_bg(result, colors, bg_idx)
        return result

    # ── axis-aligned pixelation ────────────────────────────────────────────────

    def _axis_pixelate(self, img, pal_bgr, cell_size, W, H,
                       num_levels, level_scale):
        """
        Multi-level pixelation: each level uses a larger cell (cell * scale^k).
        Coarser levels are blended over finer ones — produces the nested-block
        look of MARPAT.
        """
        cw = max(1, _nearest_divisor(W, cell_size))
        ch = max(1, _nearest_divisor(H, cell_size))
        result = _snap_to_palette(_pixelate(img, cw, ch), pal_bgr)

        for level in range(1, num_levels):
            cw2 = max(1, _nearest_divisor(W, max(1, int(cell_size * level_scale**level))))
            ch2 = max(1, _nearest_divisor(H, max(1, int(cell_size * level_scale**level))))
            coarse = _snap_to_palette(_pixelate(img, cw2, ch2), pal_bgr)
            # Only replace cells where coarse and fine disagree (adds detail)
            mask = np.any(coarse != result, axis=2)
            # Blend: ~40% of coarse overwrites fine
            blend_prob = 0.4 / level
            keep = np.random.default_rng(level).random(mask.shape) > blend_prob
            mask &= ~keep
            result[mask] = coarse[mask]

        return result

    # ── diagonal pixelation ───────────────────────────────────────────────────

    def _diagonal_pixelate(self, img, pal_bgr, cell_size, angle,
                           num_levels, level_scale, rng):
        """
        Rotate → pixelate → rotate back.
        Uses a padded canvas to reduce edge seam artefacts.
        """
        H, W   = img.shape[:2]
        pad    = int(max(W, H) * 0.15) + cell_size * 2
        # Pad with wrap to minimise seam
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="wrap")
        pH, pW = padded.shape[:2]
        pcx, pcy = pW / 2.0, pH / 2.0

        # Rotate forward
        M_fwd  = cv2.getRotationMatrix2D((pcx, pcy), angle, 1.0)
        rot    = cv2.warpAffine(padded, M_fwd, (pW, pH),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_REFLECT_101)
        # Pixelate in rotated frame
        cw = max(1, _nearest_divisor(pW, cell_size))
        ch = max(1, _nearest_divisor(pH, cell_size))
        pix = _snap_to_palette(_pixelate(rot, cw, ch), pal_bgr)

        # Rotate back
        M_bwd  = cv2.getRotationMatrix2D((pcx, pcy), -angle, 1.0)
        unrot  = cv2.warpAffine(pix, M_bwd, (pW, pH),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_REFLECT_101)
        # Crop back to original size
        result = unrot[pad:pad+H, pad:pad+W]
        if result.shape[0] != H or result.shape[1] != W:
            result = cv2.resize(result, (W, H), interpolation=cv2.INTER_NEAREST)
        return result
