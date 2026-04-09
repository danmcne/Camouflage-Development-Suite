"""
Recursive Fractal Generator – multi-scale Voronoi pyramid, memory-safe.

Depth is capped externally by passing `max_depth` in params (set by evolution
worker). Each level has an 8-second time limit; if exceeded the generator
returns what it has so far.
"""
from __future__ import annotations
import time
import threading
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS

MAX_SEEDS_PER_LEVEL = 800
BATCH        = 100
TIME_LIMIT   = 8.0

_abort_event: threading.Event = threading.Event()

def set_abort(flag: bool):
    if flag: _abort_event.set()
    else:    _abort_event.clear()


def _voronoi_layer(height, width, seeds, colors, rng, deadline):
    N       = len(seeds)
    n_c     = max(1, len(colors))
    cidx    = rng.integers(0, n_c, size=N)

    flat_x  = np.tile(np.arange(width,  dtype=np.float32), height)
    flat_y  = np.repeat(np.arange(height, dtype=np.float32), width)
    min_d   = np.full(height * width, np.inf, dtype=np.float32)
    nearest = np.zeros(height * width, dtype=np.int32)

    for start in range(0, N, BATCH):
        if _abort_event.is_set() or time.time() > deadline:
            return None
        batch = seeds[start:start + BATCH]
        dx = np.abs(flat_x[:, None] - batch[:, 0])
        dy = np.abs(flat_y[:, None] - batch[:, 1])
        dx = np.minimum(dx, width  - dx)
        dy = np.minimum(dy, height - dy)
        dist = dx * dx + dy * dy
        local_min = dist.min(axis=1)
        local_arg = dist.argmin(axis=1)
        better    = local_min < min_d
        min_d[better]    = local_min[better]
        nearest[better]  = start + local_arg[better]

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
        # Honour external depth cap (set by evolution worker)
        depth         = min(depth, int(params.get("max_depth", depth)))
        base_seeds    = int(params.get("base_seeds",      6))
        multiplier    = int(params.get("seed_multiplier", 3))
        level_opacity = float(params.get("level_opacity", 0.55))
        edge_sharp    = float(params.get("edge_sharpness",0.0))
        transparent   = bool(params.get("transparent_bg", False))
        seed          = int(params.get("seed", 42))

        rng = np.random.default_rng(seed)
        set_abort(False)

        n_seeds_0 = min(base_seeds, MAX_SEEDS_PER_LEVEL)
        seeds     = rng.uniform(0, 1, (n_seeds_0, 2)) * [width, height]

        layer = _voronoi_layer(height, width, seeds, colors, rng, time.time() + TIME_LIMIT)
        if layer is None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            if colors: r, g, b = colors[0]; canvas[:] = (b, g, r)
        else:
            canvas = layer

        for level in range(1, depth):
            if _abort_event.is_set():
                break
            n_s    = min(int(base_seeds * (multiplier ** level)), MAX_SEEDS_PER_LEVEL)
            seeds  = rng.uniform(0, 1, (n_s, 2)) * [width, height]
            layer  = _voronoi_layer(height, width, seeds, colors, rng, time.time() + TIME_LIMIT)
            if layer is None:
                break
            canvas = cv2.addWeighted(canvas, 1.0 - level_opacity, layer, level_opacity, 0)

        if edge_sharp > 0 and not _abort_event.is_set():
            blurred = cv2.GaussianBlur(canvas, (0, 0), edge_sharp)
            canvas  = cv2.addWeighted(canvas, 2.5, blurred, -1.5, 0)
            np.clip(canvas, 0, 255, out=canvas)

        if transparent:
            bgr_c = colors[0]
            mask  = np.all(canvas == (bgr_c[2], bgr_c[1], bgr_c[0]), axis=2)
            bgra  = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
            bgra[mask, 3] = 0
            return bgra

        return canvas
