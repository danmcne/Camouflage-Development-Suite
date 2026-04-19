"""
Multiscale Plaid Generator – toroidal by construction, angles 0° or 45° only.

Toroidal guarantee
──────────────────
All stripe formulas use *normalised* coordinates (xn = x/W, yn = y/H) with
*integer* period counts.  For any integer n:

  floor(xn * n) tiles seamlessly:   at x = W → xn = 1 → floor(n) = n   (wraps mod n)
  floor(yn * n) tiles seamlessly:   at y = H → yn = 1 → floor(n) = n   (same)
  floor((xn+yn)*n)  for even n:     at x = W → shifts by n → same mod-2 parity  ✓
  floor(((xn-yn)%1)*n): proven seamless for any n (see dazzle module).

No work-size, no rounding needed – the normalised formula is exact for any W×H.

Angles
──────
  0°  – axis-aligned H+V stripes.  Classic tartan / fabric plaid.
  45° – diagonal +45° and -45° stripes (two crossing diagonals).

Scale
─────
  n_base       integer number of coarse stripe periods across the canvas.
  scale_factor integer (2, 3, or 4): each finer octave = previous × scale_factor.
  num_scales   how many octaves.

Intersection colour
───────────────────
Where H and V (or D+ and D-) stripes cross, a third palette colour is
composited with reduced opacity, giving the characteristic darker/lighter
crossing zone of a woven plaid.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator, get_bg_params, apply_transparent_bg
from config.defaults import GENERATORS


# ── Blend helper ──────────────────────────────────────────────────────────────

def _blend(base_f: np.ndarray, color_bgr: np.ndarray,
           alpha: np.ndarray, mode: str) -> np.ndarray:
    """
    Alpha-composite a solid colour layer onto `base_f` (float32 BGR, [0,1]).
    `alpha` shape: (H, W, 1).  `color_bgr` shape: (3,) float.
    """
    layer = color_bgr[np.newaxis, np.newaxis, :]  # broadcast to (H,W,3)
    if   mode == "multiply": blended = base_f * layer
    elif mode == "screen":   blended = 1.0 - (1.0 - base_f) * (1.0 - layer)
    elif mode == "overlay":  blended = np.where(base_f < 0.5,
                                                2 * base_f * layer,
                                                1.0 - 2*(1-base_f)*(1-layer))
    else:                    blended = layer       # normal
    return base_f * (1.0 - alpha) + blended * alpha


# ── Soft stripe alpha ─────────────────────────────────────────────────────────

def _soft_stripe(phase: np.ndarray, softness: float) -> np.ndarray:
    """
    phase ∈ [0, 1): normalised position within a period.
    Returns float32 alpha in [0,1].  Duty = 0.5 (symmetric).
    softness = 0 → hard edge;  softness = 0.4 → very soft gradient.
    """
    # Signed distance to nearest edge in the half-period centred on 0/0.5
    d = np.minimum(phase, 1.0 - phase)          # dist to period boundary
    d = np.abs(d - 0.25) - 0.25 + softness * 0.5
    if softness < 1e-4:
        return (phase < 0.5).astype(np.float32)
    return np.clip(d / (softness + 1e-8) + 0.5, 0.0, 1.0).astype(np.float32)


# ── Stripe phase helpers ──────────────────────────────────────────────────────

def _phase_axis(xn_or_yn: np.ndarray, n: int) -> np.ndarray:
    """Normalised phase in [0,1) for axis-aligned stripe: (coord*n)%1."""
    return (xn_or_yn * n) % 1.0


def _phase_diag_plus(xn: np.ndarray, yn: np.ndarray, n: int) -> np.ndarray:
    """Phase for diagonal+ stripe ((xn+yn)*n)%1. n enforced even."""
    n_even = n + (n % 2)
    return ((xn + yn) * n_even) % 1.0


def _phase_diag_minus(xn: np.ndarray, yn: np.ndarray, n: int) -> np.ndarray:
    """Phase for diagonal- stripe: seamless for any n."""
    return ((xn - yn) % 1.0 * n) % 1.0


# ── Generator ─────────────────────────────────────────────────────────────────

class PlaidGenerator(BaseGenerator):
    name = "Plaid"
    description = (
        "Multiscale plaid — nested H+V (0°) or diagonal (45°) stripe octaves. "
        "Perfectly seamless by construction: normalised coordinates with integer "
        "period counts guarantee toroidal tiling on any canvas."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["plaid"]

    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],
        params: dict,
    ) -> np.ndarray:

        angle        = int(str(params.get("angle", 0)))          # 0 or 45 (str from combo or int)
        n_base       = max(1, int(params.get("n_base",  4)))    # coarse periods
        num_scales   = max(1, int(params.get("num_scales", 3)))
        scale_factor = max(2, int(params.get("scale_factor", 2)))
        edge_softness= float(params.get("edge_softness", 0.15))
        h_weight     = float(params.get("h_weight",     0.5))
        blend_mode   = params.get("blend_mode",        "normal")
        transparent  = bool(params.get("transparent_bg", False))
        seed         = int(params.get("seed",          42))

        rng  = np.random.default_rng(seed)
        n    = max(1, len(colors))
        W, H = width, height

        # Background: palette[0]
        bg_r, bg_g, bg_b = colors[0]
        canvas = np.full((H, W, 3),
                         [bg_b / 255.0, bg_g / 255.0, bg_r / 255.0],
                         dtype=np.float32)

        # Normalised grids – exact toroidal by construction
        ys_grid, xs_grid = np.mgrid[0:H, 0:W]
        xn = xs_grid.astype(np.float32) / W    # [0, 1)
        yn = ys_grid.astype(np.float32) / H    # [0, 1)

        base_opacity = 0.80
        color_idx    = 1        # start at palette[1] so bg stays as bg

        for scale_i in range(num_scales):
            n_i = n_base * (scale_factor ** scale_i)

            # Opacity decreases for finer scales (coarser is more opaque)
            opacity_i = base_opacity / math.sqrt(scale_i + 1)

            # Phase offset per scale so bands don't all align at origin
            phase_offset = float(rng.uniform(0.0, 1.0))

            # ── Stripe axis A (vertical in 0°, diagonal+ in 45°) ─────────────
            if angle == 45:
                phase_a = (_phase_diag_plus(xn, yn, n_i) + phase_offset) % 1.0
            else:
                phase_a = (_phase_axis(xn, n_i) + phase_offset) % 1.0

            alpha_a = _soft_stripe(phase_a, edge_softness)

            # ── Stripe axis B (horizontal in 0°, diagonal- in 45°) ───────────
            if angle == 45:
                phase_b = (_phase_diag_minus(xn, yn, n_i) + phase_offset * 0.7) % 1.0
            else:
                phase_b = (_phase_axis(yn, n_i) + phase_offset * 0.7) % 1.0

            alpha_b = _soft_stripe(phase_b, edge_softness)

            # ── Colour assignment ─────────────────────────────────────────────
            ca = colors[color_idx % n]; color_idx += 1
            cb = colors[color_idx % n]; color_idx += 1
            ci = colors[color_idx % n]; color_idx += 1      # intersection

            ca_bgr = np.array([ca[2]/255.0, ca[1]/255.0, ca[0]/255.0], np.float32)
            cb_bgr = np.array([cb[2]/255.0, cb[1]/255.0, cb[0]/255.0], np.float32)
            ci_bgr = np.array([ci[2]/255.0, ci[1]/255.0, ci[0]/255.0], np.float32)

            # Axis A (vertical / diag+)
            wa = np.clip(alpha_a[:, :, np.newaxis]
                         * (1.0 - h_weight) * 2.0 * opacity_i, 0.0, 1.0)
            canvas = _blend(canvas, ca_bgr, wa, blend_mode)

            # Axis B (horizontal / diag-)
            wb = np.clip(alpha_b[:, :, np.newaxis]
                         * h_weight * 2.0 * opacity_i, 0.0, 1.0)
            canvas = _blend(canvas, cb_bgr, wb, blend_mode)

            # Intersection: both axes active simultaneously
            wi = np.clip((alpha_a * alpha_b)[:, :, np.newaxis]
                         * opacity_i * 0.6, 0.0, 1.0)
            canvas = _blend(canvas, ci_bgr, wi, blend_mode)

        result_bgr = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)

        if transparent:
            n = max(1, len(colors))
            bg_idx, _ = get_bg_params(params, n)
            return apply_transparent_bg(result_bgr, colors, bg_idx)

        return result_bgr
