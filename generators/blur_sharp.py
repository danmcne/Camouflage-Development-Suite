"""
Blur-Sharp Generator – iterative anisotropic Gaussian blur + unsharp mask.

σX >> σY  → vertical stripes
σY >> σX  → horizontal stripes
σX == σY  → spots / labyrinth (depending on noise_amp)

All blur passes use np.pad(mode='wrap') for toroidal (seamless) convolution.
"""
from __future__ import annotations
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS


def _toroidal_gaussian_1d(field: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    """Toroidal 1-D Gaussian blur along `axis`."""
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
    """Toroidal anisotropic 2-D Gaussian via two separable passes."""
    out = _toroidal_gaussian_1d(field, sx, axis=1)
    out = _toroidal_gaussian_1d(out,   sy, axis=0)
    return out


def _unsharp(field: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Toroidal unsharp mask."""
    blurred = _toroidal_blur_2d(field, sigma, sigma)
    return field + amount * (field - blurred)


class BlurSharpGenerator(BaseGenerator):
    name = "Blur-Sharp"
    description = (
        "Iterative anisotropic blur + strong unsharp mask on seamless noise. "
        "σX ≠ σY → directional stripes (σX>>σY=vertical, σY>>σX=horizontal). "
        "Seamless toroidal output."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["blur_sharp"]

    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],
        params: dict,
    ) -> np.ndarray:
        import noise as noise_lib

        sigma_x     = float(params.get("sigma_x",       4.0))
        sigma_y     = float(params.get("sigma_y",       4.0))
        iterations  = int(params.get("iterations",      12))
        sharpen_amt = float(params.get("sharpen_amt",   4.0))
        sharpen_sig = float(params.get("sharpen_sigma", 2.0))
        noise_freq  = int(params.get("noise_freq",      4))
        noise_amp   = float(params.get("noise_amp",     0.5))
        work_size   = int(params.get("work_size",       256))
        post_blur   = float(params.get("post_blur",     1.0))
        color_mode  = params.get("color_mode",          "threshold")
        transparent = bool(params.get("transparent_bg", False))
        seed        = int(params.get("seed",            42))

        rng    = np.random.default_rng(seed)
        base_x = int(rng.integers(0, 500)) * noise_freq
        base_y = int(rng.integers(0, 500)) * noise_freq

        # ── 1. Seamless initial noise ─────────────────────────────────────────
        field = np.zeros((work_size, work_size), dtype=np.float32)
        for y in range(work_size):
            ny = (y / work_size) * noise_freq + base_y
            for x in range(work_size):
                nx = (x / work_size) * noise_freq + base_x
                field[y, x] = noise_lib.pnoise2(
                    nx, ny,
                    octaves=1,
                    repeatx=noise_freq,
                    repeaty=noise_freq,
                )
        # Normalise to [0, noise_amp]
        mn, mx = field.min(), field.max()
        if mx > mn:
            field = (field - mn) / (mx - mn) * noise_amp

        # ── 2. Iterative blur-sharpen ─────────────────────────────────────────
        for _ in range(iterations):
            field = _toroidal_blur_2d(field, sigma_x, sigma_y)
            field = _unsharp(field, sharpen_sig, sharpen_amt)
            mn, mx = field.min(), field.max()
            if mx > mn:
                field = (field - mn) / (mx - mn)
            else:
                field[:] = 0.5

        # ── 3. Upscale ────────────────────────────────────────────────────────
        if work_size != width or work_size != height:
            u8    = (field * 255).clip(0, 255).astype(np.uint8)
            big   = cv2.resize(u8, (width, height), interpolation=cv2.INTER_LINEAR)
            field = big.astype(np.float32) / 255.0

        if post_blur > 0.1:
            field = _toroidal_blur_2d(field, post_blur, post_blur)
            mn, mx = field.min(), field.max()
            if mx > mn:
                field = (field - mn) / (mx - mn)

        # ── 4. Colourise ──────────────────────────────────────────────────────
        return _colorise(field, colors, color_mode, transparent)


# ── shared colour-mapping helper ──────────────────────────────────────────────

def _colorise(field, colors, color_mode, transparent):
    h, w = field.shape
    n    = max(1, len(colors))

    if transparent:
        canvas    = np.zeros((h, w, 4), dtype=np.uint8)
        fg_colors = colors[1:] if n > 1 else colors
        nf        = max(1, len(fg_colors))
        thresholds = np.linspace(0.0, 1.0 + 1e-6, nf + 1)
        if color_mode == "gradient":
            lut = _make_lut(fg_colors)
            idx = (field * 255).clip(0, 255).astype(np.uint8)
            canvas[:, :, :3] = lut[idx]
            canvas[:, :,  3] = 255
        else:
            for i, (r, g, b) in enumerate(fg_colors):
                mask = (field >= thresholds[i]) & (field < thresholds[i + 1])
                canvas[mask] = (int(b), int(g), int(r), 255)
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if color_mode == "gradient":
            lut    = _make_lut(colors)
            idx    = (field * 255).clip(0, 255).astype(np.uint8)
            canvas = lut[idx]
        else:
            thresholds = np.linspace(0.0, 1.0 + 1e-6, n + 1)
            for i, (r, g, b) in enumerate(colors):
                mask = (field >= thresholds[i]) & (field < thresholds[i + 1])
                canvas[mask] = (int(b), int(g), int(r))

    return canvas


def _make_lut(colors):
    """256-entry BGR LUT interpolating across the colour list."""
    n   = len(colors)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t  = i / 255.0 * (n - 1)
        lo = int(t); hi = min(lo + 1, n - 1)
        a  = t - lo
        lut[i] = (
            int(colors[lo][2] * (1 - a) + colors[hi][2] * a),
            int(colors[lo][1] * (1 - a) + colors[hi][1] * a),
            int(colors[lo][0] * (1 - a) + colors[hi][0] * a),
        )
    return lut
