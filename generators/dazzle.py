"""
Dazzle Camouflage Generator – toroidal Voronoi zones with per-zone stripe patterns.

Transparency approach
─────────────────────
When transparent_bg=True, every zone draws stripe-A in the zone's fg colour
and stripe-B in colors[bg_color_idx] (the same bg colour for every zone).
After rendering, all pixels matching that bg colour are set to alpha=0.
This gives a consistent "hole" colour across the whole image — the gaps
between the fg stripes are all transparent, making the layer compositable.

When transparent_bg=False (default), each zone independently picks two
contrasting fg colours for its two stripe bands (the original behaviour).
"""
from __future__ import annotations
import numpy as np
import cv2
from generators.base import (BaseGenerator, get_bg_params,
                              apply_transparent_bg, make_fg_colors)
from config.defaults import GENERATORS


# ── Toroidal Voronoi ─────────────────────────────────────────────────────────

def _toroidal_voronoi(seeds: np.ndarray, W: int, H: int) -> np.ndarray:
    offsets  = [(dx, dy) for dx in (-W, 0, W) for dy in (-H, 0, H)]
    ext_pts  = []
    ext_ids  = []
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
        _, idx = cKDTree(ext_pts).query(pix_pts, k=1, workers=-1)
    except ImportError:
        diff = pix_pts[:, None, :] - ext_pts[None, :, :]
        idx  = np.argmin((diff**2).sum(axis=2), axis=1)
    return ext_ids[idx].reshape(H, W)


# ── Stripe formulas ───────────────────────────────────────────────────────────

_ALL_DIRS = ["vertical", "horizontal", "diagonal+", "diagonal-"]

def _stripe_binary(xn, yn, direction, n):
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

def _draw_zone_outlines(canvas, zone_map, outline_width):
    if outline_width <= 0:
        return
    z8    = (zone_map % 256).astype(np.uint8)
    edges = cv2.Canny(z8, 50, 150)
    if outline_width > 1:
        edges = cv2.dilate(edges, np.ones((outline_width, outline_width), np.uint8))
    canvas[edges > 0, :3] = 0
    if canvas.shape[2] == 4:
        canvas[edges > 0, 3] = 255


# ── Generator ─────────────────────────────────────────────────────────────────

class DazzleGenerator(BaseGenerator):
    name = "Dazzle"
    description = (
        "WWI Razzle-Dazzle: toroidal Voronoi zones each with its own stripe "
        "direction and density. When transparent_bg is on, the bg colour fills "
        "one stripe band in every zone and is then made transparent — so all "
        "gaps are the same colour and the layer composites cleanly."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["dazzle"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        num_zones      = max(3,  int(params.get("num_zones",       12)))
        stripe_density = max(2,  int(params.get("stripe_density",   8)))
        density_var    = float(params.get("density_variation",   0.5))
        directions     = params.get("directions",               "mixed")
        outline_width  = int(params.get("outline_width",          2))
        high_contrast  = bool(params.get("high_contrast",        True))
        transparent    = bool(params.get("transparent_bg",       False))
        seed           = int(params.get("seed",                  42))

        rng = np.random.default_rng(seed)
        n   = max(1, len(colors))
        bg_idx, exclude = get_bg_params(params, n)
        fg_colors = make_fg_colors(colors, bg_idx, exclude)
        nf = max(1, len(fg_colors))
        bg_color = colors[bg_idx]   # used as stripe-B when transparent
        W, H = width, height

        # ── Zones ─────────────────────────────────────────────────────────────
        seeds    = np.column_stack([rng.uniform(0,W,num_zones), rng.uniform(0,H,num_zones)])
        zone_map = _toroidal_voronoi(seeds, W, H)

        # ── Per-zone stripe params ─────────────────────────────────────────────
        dir_pool = {"mixed": _ALL_DIRS, "striped": ["vertical","horizontal"],
                    "diagonal": ["diagonal+","diagonal-"]}.get(directions, _ALL_DIRS)
        zone_dirs = [dir_pool[int(rng.integers(0, len(dir_pool)))] for _ in range(num_zones)]
        zone_ns   = [max(2, stripe_density + int(rng.integers(-max(0,int(density_var*stripe_density)),
                                                               max(0,int(density_var*stripe_density))+1)))
                     for _ in range(num_zones)]

        # ── Colour pairs per zone ──────────────────────────────────────────────
        # When transparent: stripe-A = zone fg colour, stripe-B = bg_color (same everywhere)
        # When opaque:      both stripes pick contrasting fg colours
        zone_ca = []
        zone_cb = []
        for z in range(num_zones):
            if transparent:
                # stripe-A: a fg colour; stripe-B: always bg_color (will become transparent)
                ci = (z * 2 if high_contrast else z) % nf
                zone_ca.append(fg_colors[ci])
                zone_cb.append(bg_color)
            else:
                if high_contrast and nf >= 2:
                    ci0 = (z * 2)     % nf
                    ci1 = (z * 2 + 1) % nf
                else:
                    ci0 = z % nf
                    ci1 = (z + 1) % nf
                zone_ca.append(fg_colors[ci0])
                zone_cb.append(fg_colors[ci1])

        # ── Normalised grids ───────────────────────────────────────────────────
        ys_grid, xs_grid = np.mgrid[0:H, 0:W]
        xn = xs_grid.astype(np.float32) / W
        yn = ys_grid.astype(np.float32) / H

        # ── Render ────────────────────────────────────────────────────────────
        # Always render BGR first; apply transparency afterwards
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for z in range(num_zones):
            zmask = (zone_map == z)
            if not zmask.any(): continue
            ab = _stripe_binary(xn, yn, zone_dirs[z], zone_ns[z])
            r0,g0,b0 = zone_ca[z]; r1,g1,b1 = zone_cb[z]
            canvas[zmask & ab]  = (int(b0), int(g0), int(r0))
            canvas[zmask & ~ab] = (int(b1), int(g1), int(r1))

        _draw_zone_outlines(np.dstack([canvas, np.full((H,W),255,dtype=np.uint8)]),
                            zone_map, outline_width)
        # Re-apply outline to BGR canvas (outlines function works on 4-ch copy)
        # Simpler: just detect edges and zero BGR directly
        if outline_width > 0:
            z8 = (zone_map % 256).astype(np.uint8)
            edges = cv2.Canny(z8, 50, 150)
            if outline_width > 1:
                edges = cv2.dilate(edges, np.ones((outline_width,outline_width),np.uint8))
            canvas[edges > 0] = 0

        if transparent:
            return apply_transparent_bg(canvas, colors, bg_idx)

        return canvas
