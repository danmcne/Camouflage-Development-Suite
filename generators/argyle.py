"""
Argyle Generator – seamlessly tiling diamond/argyle patterns.

Diamond count and ratio
───────────────────────
  na = n_base * ratio_h   (diamonds in the u-diagonal direction)
  nb = n_base * ratio_v   (diamonds in the v-diagonal direction)

Both na and nb are integers set directly from the UI.

Colour seamlessness
────────────────────
Colour formula: fg[(floor(u) + floor(v)) % 2]
  At x tile boundary: Δ(floor(u)+floor(v)) = na + na = 2·na ≡ 0 (mod 2) ✓
  At y tile boundary: Δ(floor(u)+floor(v)) = nb − nb = 0             ✓

So colour is ALWAYS seamless for ANY na, nb, regardless of palette size.
Two alternating foreground colours are used for the diamonds; additional
palette colours appear in the overlay lines and scale overlays.

Coordinates
───────────
  u(xn, yn) = xn·na + yn·nb    (mod 1 per period)
  v(xn, yn) = xn·na − yn·nb

The diamond is the region where BOTH stripe families are active
simultaneously.  Stripe width = 0.5 of each period → classic argyle.
"""
from __future__ import annotations
import math
import numpy as np
from generators.base import BaseGenerator, get_bg_params, apply_transparent_bg
from config.defaults import GENERATORS


def _stripe_alpha(phase: np.ndarray, softness: float) -> np.ndarray:
    """1 inside stripe (phase in [0, 0.5)), 0 in gap. Smooth at both boundaries."""
    p = phase % 1.0
    if softness < 1e-4:
        return (p < 0.5).astype(np.float32)
    a1 = np.clip((0.5 - (p - 0.5)) / softness, 0.0, 1.0)
    a2 = np.clip((0.5 - ((p + 0.5) % 1.0 - 0.5)) / softness, 0.0, 1.0)
    return np.maximum(a1, a2).astype(np.float32)


def _parse_ratio(ratio_str: str) -> tuple[int, int]:
    """'2:1 (tall x2)' → (2, 1)."""
    token = str(ratio_str).split()[0]
    parts = token.split(":")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 1
    except (ValueError, IndexError):
        return 1, 1


class ArgyleGenerator(BaseGenerator):
    name = "Argyle"
    description = (
        "Perfectly seamless argyle diamond pattern. "
        "Diamond count controls exactly how many diamonds appear across and down. "
        "Colour formula uses mod-2 alternation — always seamless for any palette."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["argyle"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        ratio_str     = str(params.get("diamond_ratio", "1:1 (square)"))
        ratio_h, ratio_v = _parse_ratio(ratio_str)
        n_base        = max(1, int(params.get("n_base", 4)))
        num_scales    = max(1, int(params.get("num_scales", 1)))
        edge_softness = float(params.get("edge_softness", 0.15))
        show_lines    = bool(params.get("show_lines", True))
        line_width    = float(params.get("line_width", 0.03))
        transparent   = bool(params.get("transparent_bg", False))
        seed          = int(params.get("seed", 42))

        rng    = np.random.default_rng(seed)
        n      = max(1, len(colors))
        bg_idx, _ = get_bg_params(params, n)
        W, H   = width, height

        ys_grid, xs_grid = np.mgrid[0:H, 0:W]
        xn = xs_grid.astype(np.float64) / W
        yn = ys_grid.astype(np.float64) / H

        bg_r, bg_g, bg_b = colors[bg_idx]
        canvas = np.full((H, W, 3),
                         [bg_b / 255.0, bg_g / 255.0, bg_r / 255.0],
                         dtype=np.float32)

        # All foreground colours (background excluded)
        fg_colors = [c for i, c in enumerate(colors) if i != bg_idx]
        if not fg_colors:
            fg_colors = colors[:]
        nfg = len(fg_colors)
        # Seed randomises starting colour index
        ci_start = int(rng.integers(0, nfg))

        for scale_i in range(num_scales):
            # na and nb set directly — no hidden multiplication
            na = n_base * ratio_h * (2 ** scale_i)
            nb = n_base * ratio_v * (2 ** scale_i)
            opacity = 0.90 / math.sqrt(scale_i + 1)

            u = xn * na + yn * nb
            v = xn * na - yn * nb

            alpha_u = _stripe_alpha(u, edge_softness)
            alpha_v = _stripe_alpha(v, edge_softness)
            diamond_alpha = alpha_u * alpha_v

            # Colour index: (floor(u) + floor(v)) % 2
            # Δx = 2·na ≡ 0 (mod 2) ✓   Δy = 0 ✓  → always seamless
            band_idx = (np.floor(u).astype(int) + np.floor(v).astype(int)) % 2

            c0r, c0g, c0b = fg_colors[ci_start % nfg]
            c1r, c1g, c1b = fg_colors[(ci_start + 1) % nfg]
            color_img = np.where(
                band_idx[:, :, np.newaxis] == 0,
                np.array([c0b / 255.0, c0g / 255.0, c0r / 255.0], np.float32),
                np.array([c1b / 255.0, c1g / 255.0, c1r / 255.0], np.float32),
            ).astype(np.float32)

            a3 = (diamond_alpha * opacity)[:, :, np.newaxis]
            canvas = canvas * (1.0 - a3) + color_img * a3

            if show_lines and scale_i == 0 and line_width > 1e-4:
                lw_frac = min(line_width, 0.48)
                line_mask = (
                    (u % 1.0 < lw_frac) | (u % 1.0 > 1.0 - lw_frac) |
                    (v % 1.0 < lw_frac) | (v % 1.0 > 1.0 - lw_frac)
                ).astype(np.float32)
                lc_idx = (ci_start + 2) % nfg
                lr, lgc, lb = fg_colors[lc_idx]
                lc_arr = np.full((H, W, 3), [lb/255.0, lgc/255.0, lr/255.0],
                                 dtype=np.float32)
                la3 = (line_mask * 0.85)[:, :, np.newaxis]
                canvas = canvas * (1.0 - la3) + lc_arr * la3

        result = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)
        if transparent:
            return apply_transparent_bg(result, colors, bg_idx)
        return result
