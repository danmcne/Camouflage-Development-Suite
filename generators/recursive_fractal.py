"""
Recursive Fractal Generator – multi-scale toroidal Voronoi pyramid.

Organic Voronoi (giraffe-like)
──────────────────────────────
Before computing Voronoi distances each pixel's coordinates are warped by a
Perlin domain-warp field.  The warp field itself uses pnoise2 with integer
repeat so it tiles seamlessly.  The result is rounded, irregular blobs that
look like giraffe patches or river stones rather than angular Voronoi cells.

warp_strength: 0 = standard angular Voronoi; 0.5–1.5 = organic blobs
warp_scale   : spatial scale of the warp noise (fraction of canvas)

Outline drawing
───────────────
outline_width > 0 draws a dark border around each cell.  We detect cell
boundaries by comparing each pixel to its right and bottom neighbours, then
dilate by outline_width pixels.
"""
from __future__ import annotations
import math
import time
import threading
import numpy as np
import cv2
from generators.base import BaseGenerator, get_bg_params, apply_transparent_bg, make_fg_colors
from config.defaults import GENERATORS

MAX_SEEDS_PER_LEVEL = 800
BATCH      = 100
TIME_LIMIT = 8.0

_abort_event: threading.Event = threading.Event()

def set_abort(flag: bool):
    if flag: _abort_event.set()
    else:    _abort_event.clear()


# ── warp field ────────────────────────────────────────────────────────────────

def _build_warp_field(height: int, width: int,
                      warp_strength: float, warp_scale: float,
                      seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return (dx, dy) float32 displacement arrays shape (H, W), or None if
    warp_strength == 0 or the noise library is unavailable.
    """
    if warp_strength < 0.01:
        return None
    try:
        import noise as noise_lib
    except ImportError:
        return None

    rng = np.random.default_rng(seed + 7777)
    periods = max(2, int(round(1.0 / max(warp_scale, 0.05))))
    bx = float(rng.uniform(0, periods))
    by = float(rng.uniform(0, periods))

    amp = warp_strength * min(width, height) * 0.25
    dx  = np.empty((height, width), dtype=np.float32)
    dy  = np.empty((height, width), dtype=np.float32)

    for iy in range(height):
        ny = (iy / height) * periods
        for ix in range(width):
            nx = (ix / width) * periods
            dx[iy, ix] = noise_lib.pnoise2(nx + bx,      ny + by,
                                            octaves=3, persistence=0.5,
                                            repeatx=periods, repeaty=periods)
            dy[iy, ix] = noise_lib.pnoise2(nx + bx + 31, ny + by + 17,
                                            octaves=3, persistence=0.5,
                                            repeatx=periods, repeaty=periods)
    return dx * amp, dy * amp


# ── Voronoi layer ─────────────────────────────────────────────────────────────

def _voronoi_layer(height, width, seeds, colors, rng, deadline,
                   warp=None):
    """
    Compute one toroidal Voronoi layer, optionally with domain warp.
    warp: (dx, dy) arrays shape (H,W) or None.
    """
    N    = len(seeds)
    n_c  = max(1, len(colors))
    cidx = rng.integers(0, n_c, size=N)

    flat_x = np.tile(np.arange(width,  dtype=np.float32), height)
    flat_y = np.repeat(np.arange(height, dtype=np.float32), width)

    if warp is not None:
        dx_flat = warp[0].ravel()
        dy_flat = warp[1].ravel()
        qx = np.clip(flat_x + dx_flat, 0, width  - 1)
        qy = np.clip(flat_y + dy_flat, 0, height - 1)
    else:
        qx, qy = flat_x, flat_y

    min_d  = np.full(height * width, np.inf, dtype=np.float32)
    nearest= np.zeros(height * width, dtype=np.int32)

    for start in range(0, N, BATCH):
        if _abort_event.is_set() or time.time() > deadline:
            return None
        batch = seeds[start:start+BATCH]
        dx    = np.abs(qx[:, None] - batch[:, 0])
        dy    = np.abs(qy[:, None] - batch[:, 1])
        dx    = np.minimum(dx, width  - dx)
        dy    = np.minimum(dy, height - dy)
        dist  = dx*dx + dy*dy
        lm    = dist.min(axis=1); la = dist.argmin(axis=1)
        better= lm < min_d
        min_d[better]    = lm[better]
        nearest[better]  = start + la[better]

    nearest = nearest.reshape(height, width)
    layer   = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = colors[cidx[i]]
        layer[nearest == i] = (int(b), int(g), int(r))
    return layer, nearest, cidx


def _draw_outlines(canvas: np.ndarray, label_map: np.ndarray,
                   outline_width: int,
                   outline_color: tuple = (0, 0, 0)) -> None:
    """Draw cell borders in-place (BGR or BGRA canvas) using the given BGR colour."""
    if outline_width <= 0:
        return
    diff_r = (label_map[:, :-1] != label_map[:, 1:])
    diff_d = (label_map[:-1, :] != label_map[1:, :])
    border  = np.zeros(label_map.shape, dtype=np.uint8)
    border[:, :-1][diff_r] = 255
    border[:-1, :][diff_d] = 255
    if outline_width > 1:
        kernel = np.ones((outline_width, outline_width), np.uint8)
        border = cv2.dilate(border, kernel)
    b, g, r = int(outline_color[0]), int(outline_color[1]), int(outline_color[2])
    canvas[border > 0, 0] = b
    canvas[border > 0, 1] = g
    canvas[border > 0, 2] = r
    if canvas.shape[2] == 4:
        canvas[border > 0, 3] = 255


# ── generator ─────────────────────────────────────────────────────────────────

class RecursiveFractalGenerator(BaseGenerator):
    name = "Recursive Fractal"
    description = (
        "Multi-scale toroidal Voronoi pyramid. "
        "Organic warp (warp_strength > 0) produces giraffe-patch shapes. "
        "outline_width draws cell borders. Seeds capped at 800/level."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["recursive_fractal"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        depth         = int(params.get("depth",           3))
        depth         = min(depth, int(params.get("max_depth", depth)))
        base_seeds    = int(params.get("base_seeds",      6))
        multiplier    = int(params.get("seed_multiplier", 3))
        level_opacity = float(params.get("level_opacity", 0.55))
        edge_sharp    = float(params.get("edge_sharpness",0.0))
        warp_strength = float(params.get("warp_strength", 0.0))
        warp_scale    = float(params.get("warp_scale",    0.3))
        outline_width = int(params.get("outline_width",   0))
        transparent   = bool(params.get("transparent_bg", False))
        seed          = int(params.get("seed",            42))

        rng = np.random.default_rng(seed)
        set_abort(False)

        n = max(1, len(colors))
        bg_idx, exclude = get_bg_params(params, n)
        # When transparent, we need bg colour to appear in some cells so
        # apply_transparent_bg can detect and zero them.
        # Only exclude bg from elements when NOT transparent.
        fg_colors = make_fg_colors(colors, bg_idx, exclude and not transparent)

        warp = _build_warp_field(height, width, warp_strength, warp_scale, seed)

        n_seeds_0 = min(base_seeds, MAX_SEEDS_PER_LEVEL)
        seeds     = rng.uniform(0, 1, (n_seeds_0, 2)) * [width, height]

        result = _voronoi_layer(height, width, seeds, fg_colors, rng,
                                time.time() + TIME_LIMIT, warp)
        if result is None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            r, g, b = colors[bg_idx]
            canvas[:] = (int(b), int(g), int(r))
            label_map = np.zeros((height, width), dtype=np.int32)
        else:
            canvas, label_map, _cidx_top = result

        for level in range(1, depth):
            if _abort_event.is_set(): break
            n_s   = min(int(base_seeds * (multiplier ** level)), MAX_SEEDS_PER_LEVEL)
            seeds = rng.uniform(0, 1, (n_s, 2)) * [width, height]
            res2  = _voronoi_layer(height, width, seeds, fg_colors, rng,
                                   time.time() + TIME_LIMIT, warp)
            if res2 is None: break
            layer, _, _cidx2 = res2
            canvas = cv2.addWeighted(canvas, 1.0 - level_opacity, layer, level_opacity, 0)

        if edge_sharp > 0 and not _abort_event.is_set():
            blurred = cv2.GaussianBlur(canvas, (0, 0), edge_sharp)
            canvas  = cv2.addWeighted(canvas, 2.5, blurred, -1.5, 0)
            np.clip(canvas, 0, 255, out=canvas)

        # Draw outlines on final composited image
        if outline_width > 0 and label_map is not None:
            r0, g0, b0 = colors[bg_idx]
            outline_bgr = (int(b0), int(g0), int(r0))
            canvas4 = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
            _draw_outlines(canvas4, label_map, outline_width, outline_bgr)
            canvas  = cv2.cvtColor(canvas4, cv2.COLOR_BGRA2BGR)

        if transparent:
            # addWeighted blending destroys exact colour values so exact-pixel
            # matching cannot work.  We use a separate Perlin noise field
            # (keyed to the same seed) as a threshold transparency mask —
            # the fraction of transparent pixels equals 1/n_colours.
            bgra = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
            try:
                import noise as noise_lib
                rng_t   = np.random.default_rng(seed ^ 0xDEAD)
                periods = max(2, base_seeds)
                bxt = float(rng_t.uniform(0, periods))
                byt = float(rng_t.uniform(0, periods))
                tfield = np.zeros((height, width), dtype=np.float32)
                for iy in range(height):
                    ny = (iy / height) * periods
                    for ix in range(width):
                        nx = (ix / width) * periods
                        tfield[iy, ix] = noise_lib.pnoise2(
                            nx+bxt, ny+byt, octaves=2,
                            repeatx=periods, repeaty=periods)
                mn, mx = tfield.min(), tfield.max()
                tfield = (tfield - mn) / (mx - mn + 1e-8)
                # threshold: 1/n_colours of pixels transparent
                thresh  = 1.0 / max(1, len(colors))
                bgra[tfield < thresh, 3] = 0
            except ImportError:
                # fallback: use label map if available
                if label_map is not None and _cidx_top is not None:
                    n_fg = max(1, len(fg_colors))
                    bg_ci_set = set(range(0, n_fg, max(1, len(colors))))
                    colour_idx_map = _cidx_top[label_map]
                    alpha = np.ones((height, width), dtype=np.uint8) * 255
                    for ci in bg_ci_set:
                        alpha[colour_idx_map == ci] = 0
                    bgra[:, :, 3] = alpha
            return bgra
        return canvas
