"""
Recursive Fractal Generator – memory-safe multi-scale Voronoi pyramid.

Safety measures:
  • Total seeds per level capped at MAX_SEEDS_PER_LEVEL (800).
  • Each level's distance computation is done in batches of BATCH to avoid
    allocating one giant (H×W×N) array.
  • A threading.Event abort flag lets the QThread worker cancel cleanly
    if the user presses Stop; the generator checks it between levels.
  • Hard time limit: if a level takes > TIME_LIMIT_SEC seconds, the generator
    returns what it has so far rather than crashing or hanging.
"""
from __future__ import annotations
import time
import threading
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS

MAX_SEEDS_PER_LEVEL = 800
BATCH = 100          # seeds processed at once when computing distances
TIME_LIMIT_SEC = 8.0  # per-level time limit

# Module-level abort event that the QThread worker can set
_abort_event: threading.Event = threading.Event()


def set_abort(flag: bool):
    if flag:
        _abort_event.set()
    else:
        _abort_event.clear()


def _voronoi_layer_batched(
    height: int,
    width: int,
    seeds: np.ndarray,   # (N, 2) float [x, y]
    colors: list[tuple[int, int, int]],
    rng: np.random.Generator,
    deadline: float,
) -> np.ndarray | None:
    """
    Render one Voronoi layer using batched distance computation.
    Returns None if aborted or time exceeded.
    """
    N = len(seeds)
    n_colors = max(1, len(colors))

    # Assign a palette colour to each seed
    seed_color_idx = rng.integers(0, n_colors, size=N)

    # Build grid coordinates (flat)
    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width,  dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")  # (H, W)
    flat_x = grid_x.ravel()   # (H*W,)
    flat_y = grid_y.ravel()

    min_dist   = np.full(height * width, np.inf, dtype=np.float32)
    nearest_idx= np.zeros(height * width, dtype=np.int32)

    for start in range(0, N, BATCH):
        if _abort_event.is_set() or time.time() > deadline:
            return None
        batch = seeds[start:start + BATCH]   # (b, 2)
        sx = batch[:, 0]   # (b,)
        sy = batch[:, 1]

        # Toroidal distance: (H*W, b)
        dx = np.abs(flat_x[:, np.newaxis] - sx[np.newaxis, :])
        dy = np.abs(flat_y[:, np.newaxis] - sy[np.newaxis, :])
        dx = np.minimum(dx, width  - dx)
        dy = np.minimum(dy, height - dy)
        dist = dx * dx + dy * dy   # (H*W, b)

        # Update minimum
        local_min = dist.min(axis=1)          # (H*W,)
        local_arg = dist.argmin(axis=1)       # (H*W,) index within batch
        better = local_min < min_dist
        min_dist[better]    = local_min[better]
        nearest_idx[better] = start + local_arg[better]

    nearest_idx = nearest_idx.reshape(height, width)

    # Build colour array
    layer = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(N):
        ci = seed_color_idx[i]
        r, g, b = colors[ci]
        layer[nearest_idx == i] = (int(b), int(g), int(r))

    return layer


class RecursiveFractalGenerator(BaseGenerator):
    name = "Recursive Fractal"
    description = (
        "Multi-scale Voronoi pyramid. Each level adds finer cells blended over "
        "coarser ones. Seeds capped at 800/level for safety. "
        "Toroidal (seamless) distance metric."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["recursive_fractal"]

    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],
        params: dict,
    ) -> np.ndarray:
        depth         = int(params.get("depth",           3))
        base_seeds    = int(params.get("base_seeds",      6))
        multiplier    = int(params.get("seed_multiplier", 3))
        level_opacity = float(params.get("level_opacity", 0.55))
        edge_sharp    = float(params.get("edge_sharpness",0.0))
        seed          = int(params.get("seed", 42))

        rng = np.random.default_rng(seed)
        set_abort(False)

        # ── Level 0: coarsest ─────────────────────────────────────────────────
        n_seeds_0 = min(base_seeds, MAX_SEEDS_PER_LEVEL)
        seeds = rng.uniform(0, 1, (n_seeds_0, 2)) * np.array([width, height])

        deadline = time.time() + TIME_LIMIT_SEC
        layer = _voronoi_layer_batched(height, width, seeds, colors, rng, deadline)

        if layer is None:
            # Aborted: return solid first colour
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            if colors:
                r, g, b = colors[0]
                canvas[:] = (b, g, r)
            return canvas

        canvas = layer

        # ── Finer levels ──────────────────────────────────────────────────────
        for level in range(1, depth):
            if _abort_event.is_set():
                break

            n_seeds = min(
                int(base_seeds * (multiplier ** level)),
                MAX_SEEDS_PER_LEVEL,
            )
            seeds = rng.uniform(0, 1, (n_seeds, 2)) * np.array([width, height])

            deadline = time.time() + TIME_LIMIT_SEC
            layer = _voronoi_layer_batched(height, width, seeds, colors, rng, deadline)

            if layer is None:
                break  # time exceeded – keep what we have

            canvas = cv2.addWeighted(canvas, 1.0 - level_opacity,
                                     layer,  level_opacity, 0)

        # ── Optional edge sharpening ──────────────────────────────────────────
        if edge_sharp > 0 and not _abort_event.is_set():
            blurred = cv2.GaussianBlur(canvas, (0, 0), edge_sharp)
            canvas  = cv2.addWeighted(canvas, 2.5, blurred, -1.5, 0)
            np.clip(canvas, 0, 255, out=canvas)

        return canvas
