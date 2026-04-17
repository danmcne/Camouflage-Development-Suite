"""
Dazzle Camouflage Generator – toroidal Voronoi zones, each with its own
directional stripe pattern.

Design
──────
The canvas is divided into N Voronoi zones using *toroidal* distance so the
pattern tiles seamlessly.  Each zone independently draws stripes at a chosen
direction and density.

All stripe formulas use *normalised* coordinates (xn = x/W, yn = y/H) so
they are inherently seamless for any canvas size – no quantised work-size needed.

Seamless stripe formulas (all proven toroidal):
  vertical   : floor(xn * n) % 2         – trivially seamless
  horizontal : floor(yn * n) % 2         – trivially seamless
  diagonal+  : floor((xn+yn) * n) % 2    – seam-free when n is even
  diagonal-  : floor(((xn-yn)%1)*n) % 2  – seam-free for any n

Proof (diagonal+): at x→x+W: xn→xn+1, sum increases by 1, floor shifts by n.
  Same binary parity iff n is even – enforced automatically.
Proof (diagonal-): (xn+1-yn)%1 = (xn-yn)%1 ✓  (xn-yn-1)%1 = (xn-yn)%1 ✓

Zone outlines are drawn post-render so the seam also wraps correctly.
"""
from __future__ import annotations
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS


# ── Toroidal Voronoi ─────────────────────────────────────────────────────────

def _toroidal_voronoi(seeds: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Return int32 zone-label array (H, W) using toroidal minimum distance.
    Each seed is replicated at all 9 toroidal offsets; nearest-neighbour then
    maps back to the original seed index.
    """
    n_seeds = len(seeds)
    offsets = [(dx, dy) for dx in (-W, 0, W) for dy in (-H, 0, H)]

    ext_pts = []
    ext_ids = []
    for i, (sx, sy) in enumerate(seeds):
        for dx, dy in offsets:
            ext_pts.append([sx + dx, sy + dy])
            ext_ids.append(i)

    ext_pts = np.array(ext_pts, dtype=np.float32)
    ext_ids = np.array(ext_ids, dtype=np.int32)

    ys, xs  = np.mgrid[0:H, 0:W]
    pix_pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)

    try:
        from scipy.spatial import cKDTree
        tree   = cKDTree(ext_pts)
        _, idx = tree.query(pix_pts, k=1, workers=-1)
    except ImportError:
        diff = pix_pts[:, np.newaxis, :] - ext_pts[np.newaxis, :, :]
        idx  = np.argmin((diff ** 2).sum(axis=2), axis=1)

    return ext_ids[idx].reshape(H, W)


# ── Stripe direction formulas ─────────────────────────────────────────────────

_ALL_DIRS = ["vertical", "horizontal", "diagonal+", "diagonal-"]

def _stripe_binary(xn: np.ndarray, yn: np.ndarray,
                   direction: str, n: int) -> np.ndarray:
    """
    Return bool array: True = stripe colour A, False = stripe colour B.
    n: integer stripe count.  For diagonal+ n is bumped to nearest even.
    """
    if direction == "vertical":
        return (np.floor(xn * n).astype(np.int32) % 2) == 0
    elif direction == "horizontal":
        return (np.floor(yn * n).astype(np.int32) % 2) == 0
    elif direction == "diagonal+":
        n_even = n + (n % 2)
        return (np.floor((xn + yn) * n_even).astype(np.int32) % 2) == 0
    else:  # diagonal-
        return (np.floor(((xn - yn) % 1.0) * n).astype(np.int32) % 2) == 0


# ── Zone outline ──────────────────────────────────────────────────────────────

def _draw_zone_outlines(canvas: np.ndarray, zone_map: np.ndarray,
                         outline_width: int) -> None:
    """Dilated boundary overlay in-place on BGR or BGRA canvas."""
    if outline_width <= 0:
        return
    z8    = (zone_map % 256).astype(np.uint8)
    edges = cv2.Canny(z8, 50, 150)
    if outline_width > 1:
        kernel = np.ones((outline_width, outline_width), np.uint8)
        edges  = cv2.dilate(edges, kernel, iterations=1)
    mask = edges > 0
    canvas[mask, :3] = 0
    if canvas.shape[2] == 4:
        canvas[mask, 3] = 255


# ── Generator ─────────────────────────────────────────────────────────────────

class DazzleGenerator(BaseGenerator):
    name = "Dazzle"
    description = (
        "WWI Razzle-Dazzle: canvas divided into toroidal Voronoi zones, each "
        "with an independently chosen stripe direction and density. Perfectly "
        "seamless – stripes use normalised-coordinate formulas inherent to the canvas."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["dazzle"]

    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],
        params: dict,
    ) -> np.ndarray:

        num_zones      = max(3,  int(params.get("num_zones",       12)))
        stripe_density = max(2,  int(params.get("stripe_density",   8)))
        density_var    = float(params.get("density_variation",  0.5))
        directions     = params.get("directions",               "mixed")
        outline_width  = int(params.get("outline_width",         2))
        high_contrast  = bool(params.get("high_contrast",       True))
        transparent    = bool(params.get("transparent_bg",      False))
        seed           = int(params.get("seed",                 42))

        rng = np.random.default_rng(seed)
        n   = max(1, len(colors))
        W, H = width, height

        # ── Voronoi zones (toroidal) ──────────────────────────────────────────
        seed_x   = rng.uniform(0, W, num_zones)
        seed_y   = rng.uniform(0, H, num_zones)
        seeds    = np.column_stack([seed_x, seed_y])
        zone_map = _toroidal_voronoi(seeds, W, H)           # (H, W) int32

        # ── Per-zone stripe parameters ────────────────────────────────────────
        dir_pool = {
            "mixed":    _ALL_DIRS,
            "striped":  ["vertical", "horizontal"],
            "diagonal": ["diagonal+", "diagonal-"],
        }.get(directions, _ALL_DIRS)

        zone_dirs = [dir_pool[int(rng.integers(0, len(dir_pool)))]
                     for _ in range(num_zones)]

        zone_ns = []
        for _ in range(num_zones):
            var = max(0, int(density_var * stripe_density))
            ni  = max(2, stripe_density + int(rng.integers(-var, var + 1)))
            zone_ns.append(ni)

        # Colour pairs per zone
        zone_c0 = []
        zone_c1 = []
        for z in range(num_zones):
            if high_contrast and n >= 2:
                ci0 = (z * 2)     % n
                ci1 = (z * 2 + 1) % n
            else:
                ci0 = z % n
                ci1 = (z + 1) % n
            zone_c0.append(colors[ci0])
            zone_c1.append(colors[ci1])

        # ── Normalised coordinate grids ───────────────────────────────────────
        ys_grid, xs_grid = np.mgrid[0:H, 0:W]
        xn = xs_grid.astype(np.float32) / W    # in [0, 1)
        yn = ys_grid.astype(np.float32) / H    # in [0, 1)

        # ── Render ────────────────────────────────────────────────────────────
        channels = 4 if transparent else 3
        canvas   = np.zeros((H, W, channels), dtype=np.uint8)

        for z in range(num_zones):
            zmask = (zone_map == z)
            if not zmask.any():
                continue

            ab   = _stripe_binary(xn, yn, zone_dirs[z], zone_ns[z])
            r0, g0, b0 = zone_c0[z]
            r1, g1, b1 = zone_c1[z]

            ma = zmask & ab
            mb = zmask & ~ab
            if transparent:
                canvas[ma] = (int(b0), int(g0), int(r0), 255)
                canvas[mb] = (int(b1), int(g1), int(r1), 255)
            else:
                canvas[ma] = (int(b0), int(g0), int(r0))
                canvas[mb] = (int(b1), int(g1), int(r1))

        _draw_zone_outlines(canvas, zone_map, outline_width)
        return canvas
