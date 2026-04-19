"""
Recursive Fractal Generator – multi-scale Voronoi pyramid, memory-safe.

Depth is capped externally by passing `max_depth` in params (set by evolution
worker). Each level has an 8-second time limit.

Background colour handling: bg_color_idx chooses which palette entry is the
background. If exclude_bg_from_elements=True, Voronoi cells never receive that
colour (it's only used for the canvas fill). transparent_bg makes bg-coloured
pixels transparent in the output.
"""
from __future__ import annotations
import time
import threading
import numpy as np
import cv2
from generators.base import BaseGenerator, get_bg_params, apply_transparent_bg, make_fg_colors
from config.defaults import GENERATORS

MAX_SEEDS_PER_LEVEL = 800
BATCH        = 100
TIME_LIMIT   = 8.0

_abort_event: threading.Event = threading.Event()

def set_abort(flag: bool):
    if flag: _abort_event.set()
    else:    _abort_event.clear()


def _voronoi_layer(height, width, seeds, colors, rng, deadline):
    N    = len(seeds)
    n_c  = max(1, len(colors))
    cidx = rng.integers(0, n_c, size=N)

    flat_x = np.tile(np.arange(width,  dtype=np.float32), height)
    flat_y = np.repeat(np.arange(height, dtype=np.float32), width)
    min_d  = np.full(height * width, np.inf, dtype=np.float32)
    nearest= np.zeros(height * width, dtype=np.int32)

    for start in range(0, N, BATCH):
        if _abort_event.is_set() or time.time() > deadline:
            return None
        batch  = seeds[start:start+BATCH]
        dx     = np.abs(flat_x[:, None] - batch[:, 0])
        dy     = np.abs(flat_y[:, None] - batch[:, 1])
        dx     = np.minimum(dx, width  - dx)
        dy     = np.minimum(dy, height - dy)
        dist   = dx*dx + dy*dy
        lm     = dist.min(axis=1); la = dist.argmin(axis=1)
        better = lm < min_d
        min_d[better]   = lm[better]
        nearest[better] = start + la[better]

    nearest = nearest.reshape(height, width)
    layer   = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = colors[cidx[i]]
        layer[nearest == i] = (int(b), int(g), int(r))
    return layer


class RecursiveFractalGenerator(BaseGenerator):
    name = "Recursive Fractal"
    description = (
        "Multi-scale Voronoi pyramid. Seeds capped at 800/level; "
        "each level has an 8-second timeout. Depth capped at 3 in evolution."
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
        transparent   = bool(params.get("transparent_bg", False))
        seed          = int(params.get("seed", 42))

        rng = np.random.default_rng(seed)
        set_abort(False)

        n = max(1, len(colors))
        bg_idx, exclude = get_bg_params(params, n)
        fg_colors = make_fg_colors(colors, bg_idx, exclude)

        n_seeds_0 = min(base_seeds, MAX_SEEDS_PER_LEVEL)
        seeds     = rng.uniform(0, 1, (n_seeds_0, 2)) * [width, height]

        layer = _voronoi_layer(height, width, seeds, fg_colors, rng, time.time() + TIME_LIMIT)
        if layer is None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            r, g, b = colors[bg_idx]
            canvas[:] = (int(b), int(g), int(r))
        else:
            canvas = layer

        for level in range(1, depth):
            if _abort_event.is_set():
                break
            n_s   = min(int(base_seeds * (multiplier ** level)), MAX_SEEDS_PER_LEVEL)
            seeds = rng.uniform(0, 1, (n_s, 2)) * [width, height]
            layer = _voronoi_layer(height, width, seeds, fg_colors, rng, time.time() + TIME_LIMIT)
            if layer is None:
                break
            canvas = cv2.addWeighted(canvas, 1.0 - level_opacity, layer, level_opacity, 0)

        if edge_sharp > 0 and not _abort_event.is_set():
            blurred = cv2.GaussianBlur(canvas, (0, 0), edge_sharp)
            canvas  = cv2.addWeighted(canvas, 2.5, blurred, -1.5, 0)
            np.clip(canvas, 0, 255, out=canvas)

        if transparent:
            return apply_transparent_bg(canvas, colors, bg_idx)

        return canvas
