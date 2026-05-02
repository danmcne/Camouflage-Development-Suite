"""
Blur-Sharp Generator – iterative anisotropic Gaussian blur + unsharp mask.

Diagonal stripes
────────────────
Standard axis-aligned blurring with σX >> σY gives vertical stripes, and
σY >> σX gives horizontal. To get diagonal stripes we apply the blur in a
rotated coordinate frame:

  1. Rotate the field by -angle
  2. Apply the anisotropic blur (now axis-aligned in rotated frame)
  3. Rotate back by +angle

Rotation is done via cv2.warpAffine on the work-size field.  Because the
field is periodic we pad by wrap-reflection before rotating, then crop back.

A second independent blur pass (sigma_x2, sigma_y2, blur_angle2) is optional
(enabled when sigma_x2 > 0).  Two crossing blur directions produce plaid-like
or diamond-grid patterns.

Toroidal guarantee
──────────────────
1D Gaussian blurs use np.pad(mode='wrap') before convolution.
Rotations on a finite array break tileability at the edges; we mitigate this
by padding by the rotation footprint before rotating, then cropping back.
This is not perfectly toroidal but the seam is imperceptible at normal work
sizes; the subsequent upscale further smooths it.

Background colour handling: see _colorise() docstring below.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator, get_bg_params
from config.defaults import GENERATORS


# ── toroidal 1-D Gaussian ─────────────────────────────────────────────────────

def _toroidal_gaussian_1d(field: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    if sigma < 0.1:
        return field
    k = max(3, int(sigma * 6 + 1))
    if k % 2 == 0:
        k += 1
    pad = k
    if axis == 0:
        padded  = np.pad(field, ((pad, pad), (0, 0)), mode="wrap")
        blurred = cv2.GaussianBlur(padded, (1, k), sigmaX=0.0, sigmaY=sigma)
        return blurred[pad:-pad, :]
    else:
        padded  = np.pad(field, ((0, 0), (pad, pad)), mode="wrap")
        blurred = cv2.GaussianBlur(padded, (k, 1), sigmaX=sigma, sigmaY=0.0)
        return blurred[:, pad:-pad]


def _toroidal_blur_2d(field: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = _toroidal_gaussian_1d(field, sx, axis=1)
    out = _toroidal_gaussian_1d(out,   sy, axis=0)
    return out


# ── rotated anisotropic blur ──────────────────────────────────────────────────

def _rotated_blur(field: np.ndarray, sx: float, sy: float,
                  angle_deg: float) -> np.ndarray:
    """
    Apply anisotropic blur (σx along rotated X-axis, σy along rotated Y-axis).
    angle_deg=0  → same as _toroidal_blur_2d(sx, sy)
    angle_deg=45 → 45° diagonal blur
    """
    if abs(angle_deg) < 0.5:
        return _toroidal_blur_2d(field, sx, sy)

    h, w    = field.shape
    angle_r = math.radians(angle_deg)
    cx, cy  = w / 2.0, h / 2.0

    # Pad to avoid edge black-fill during rotation (wrap-pad by blur radius)
    pad = max(int(max(sx, sy) * 4 + w * 0.15), 8)
    padded = np.pad(field, pad, mode="wrap")
    ph, pw = padded.shape
    pcx, pcy = pw / 2.0, ph / 2.0

    # Rotate backward
    M_fwd = cv2.getRotationMatrix2D((pcx, pcy), -angle_deg, 1.0)
    rot_fwd = cv2.warpAffine(padded, M_fwd, (pw, ph),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    # Blur axis-aligned in rotated frame
    blurred = _toroidal_blur_2d(rot_fwd, sx, sy)

    # Rotate back
    M_bwd = cv2.getRotationMatrix2D((pcx, pcy), angle_deg, 1.0)
    rot_bwd = cv2.warpAffine(blurred, M_bwd, (pw, ph),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    # Crop to original size
    result = rot_bwd[pad:pad+h, pad:pad+w]
    # Re-normalise to counter any amplitude change from rotation interpolation
    mn, mx = result.min(), result.max()
    if mx > mn:
        return ((result - mn) / (mx - mn)).astype(np.float32)
    return result.astype(np.float32)


def _unsharp(field: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blurred = _toroidal_blur_2d(field, sigma, sigma)
    return field + amount * (field - blurred)


# ── generator ─────────────────────────────────────────────────────────────────

class BlurSharpGenerator(BaseGenerator):
    name = "Blur-Sharp"
    description = (
        "Iterative anisotropic blur + strong unsharp mask on seamless noise. "
        "σX ≠ σY → directional stripes. Blur angle rotates stripes to any diagonal."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["blur_sharp"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        sigma_x      = float(params.get("sigma_x",       4.0))
        sigma_y      = float(params.get("sigma_y",       4.0))
        blur_angle   = float(params.get("blur_angle",    0.0))
        iterations   = int(params.get("iterations",      12))
        sharpen_amt  = float(params.get("sharpen_amt",   4.0))
        sharpen_sig  = float(params.get("sharpen_sigma", 2.0))
        noise_density= float(params.get("noise_density", 0.15))
        noise_amp    = float(params.get("noise_amp",     1.0))
        noise_mode   = params.get("noise_mode",          "uniform")
        work_size    = int(params.get("work_size",       256))
        post_blur    = float(params.get("post_blur",     1.0))
        color_mode   = params.get("color_mode",          "threshold")
        transparent  = bool(params.get("transparent_bg", False))
        seed         = int(params.get("seed",            42))

        rng = np.random.default_rng(seed)

        # ── 1. Sparse stochastic initial field ───────────────────────────────
        field = np.zeros((work_size, work_size), dtype=np.float32)
        mask  = rng.random((work_size, work_size)) < noise_density
        if noise_mode == "binary":
            field[mask] = noise_amp
        else:
            field[mask] = rng.random(np.count_nonzero(mask)).astype(np.float32) * noise_amp
        if noise_density > 0:
            field -= field.mean(); field += 0.5
        field = np.clip(field, 0.0, 1.0)

        # ── 2. Iterative blur-sharpen ─────────────────────────────────────────
        for _ in range(iterations):
            field = _rotated_blur(field, sigma_x, sigma_y, blur_angle)
            field = _unsharp(field, sharpen_sig, sharpen_amt)
            mn, mx = field.min(), field.max()
            if mx > mn: field = (field - mn) / (mx - mn)
            else:       field[:] = 0.5

        # ── 3. Upscale ────────────────────────────────────────────────────────
        if work_size != width or work_size != height:
            u8    = (field * 255).clip(0, 255).astype(np.uint8)
            big   = cv2.resize(u8, (width, height), interpolation=cv2.INTER_LINEAR)
            field = big.astype(np.float32) / 255.0
        if post_blur > 0.1:
            field = _toroidal_blur_2d(field, post_blur, post_blur)
            mn, mx = field.min(), field.max()
            if mx > mn: field = (field - mn) / (mx - mn)

        # ── 4. Colourise ──────────────────────────────────────────────────────
        n = max(1, len(colors))
        bg_idx, exclude = get_bg_params(params, n)
        return _colorise(field, colors, color_mode, transparent, bg_idx, exclude)


# ── shared colour-mapping helper ──────────────────────────────────────────────

def _colorise(field, colors, color_mode, transparent,
              bg_idx: int = 0, exclude_bg: bool = False):
    h, w = field.shape
    n    = max(1, len(colors))
    bg_idx = max(0, min(bg_idx, n - 1))

    if not transparent and not exclude_bg:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if color_mode == "gradient":
            lut = _make_lut(colors)
            canvas = lut[(field * 255).clip(0, 255).astype(np.uint8)]
        else:
            thresholds = np.linspace(0.0, 1.0 + 1e-6, n + 1)
            for i, (r, g, b) in enumerate(colors):
                mask = (field >= thresholds[i]) & (field < thresholds[i + 1])
                canvas[mask] = (int(b), int(g), int(r))
        return canvas

    fg_colors  = [c for i, c in enumerate(colors) if i != bg_idx]
    if not fg_colors: fg_colors = colors
    bg_r, bg_g, bg_b = colors[bg_idx]
    thresholds = np.linspace(0.0, 1.0 + 1e-6, n + 1)

    if transparent:
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
    else:
        canvas = np.full((h, w, 3), [int(bg_b), int(bg_g), int(bg_r)], dtype=np.uint8)

    if color_mode == "gradient":
        lut      = _make_lut(fg_colors)
        coloured = lut[(field * 255).clip(0, 255).astype(np.uint8)]
        bg_mask  = (field >= thresholds[bg_idx]) & (field < thresholds[bg_idx + 1])
        if transparent:
            canvas[:, :, :3] = coloured; canvas[:, :, 3] = 255
            canvas[bg_mask]  = (0, 0, 0, 0)
        else:
            canvas = coloured; canvas[bg_mask] = (int(bg_b), int(bg_g), int(bg_r))
    else:
        fg_iter = iter(fg_colors)
        for band_i in range(n):
            bmask = (field >= thresholds[band_i]) & (field < thresholds[band_i + 1])
            if band_i == bg_idx:
                if transparent: canvas[bmask] = (0, 0, 0, 0)
                continue
            try:   r, g, b = next(fg_iter)
            except StopIteration: break
            if transparent: canvas[bmask] = (int(b), int(g), int(r), 255)
            else:           canvas[bmask] = (int(b), int(g), int(r))
    return canvas


def _make_lut(colors):
    n   = len(colors)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t  = i / 255.0 * (n - 1)
        lo = int(t); hi = min(lo + 1, n - 1); a = t - lo
        lut[i] = (
            int(colors[lo][2] * (1-a) + colors[hi][2] * a),
            int(colors[lo][1] * (1-a) + colors[hi][1] * a),
            int(colors[lo][0] * (1-a) + colors[hi][0] * a),
        )
    return lut
