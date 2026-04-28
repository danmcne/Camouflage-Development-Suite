"""
Dazzle Camouflage Generator – toroidal Voronoi zones with per-zone stripe patterns.

Organic mode
────────────
When warp_strength > 0, a Perlin domain warp is applied to the Voronoi seed
lookup coordinates before distance computation.  This rounds the cell
boundaries into giraffe-patch / stone shapes rather than angular polygons.
The warp field uses integer-repeat pnoise2 so it tiles seamlessly.

Thick outlines
──────────────
outline_width controls the border width in pixels (0 = none).
Borders are drawn by detecting cell-boundary pixels (label differs from
right/bottom neighbour) and dilating.

Transparency
────────────
When transparent_bg=True, stripe-B in every zone is filled with colors[bg_idx]
(same colour everywhere), then apply_transparent_bg() zeroes those pixels.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import (BaseGenerator, get_bg_params,
                              apply_transparent_bg, make_fg_colors)
from config.defaults import GENERATORS


# ── Toroidal Voronoi (with optional warp) ────────────────────────────────────

def _build_warp(height, width, warp_strength, warp_scale, seed):
    if warp_strength < 0.01:
        return None
    try:
        import noise as noise_lib
    except ImportError:
        return None
    rng     = np.random.default_rng(seed + 3131)
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
            dx[iy, ix] = noise_lib.pnoise2(nx+bx,    ny+by,    octaves=3,
                                            repeatx=periods, repeaty=periods)
            dy[iy, ix] = noise_lib.pnoise2(nx+bx+31, ny+by+17, octaves=3,
                                            repeatx=periods, repeaty=periods)
    return dx * amp, dy * amp


def _toroidal_voronoi(seeds: np.ndarray, W: int, H: int,
                      warp=None) -> np.ndarray:
    """Return int32 zone-label array (H, W) using toroidal distance + optional warp."""
    offsets = [(dx, dy) for dx in (-W, 0, W) for dy in (-H, 0, H)]
    ext_pts, ext_ids = [], []
    for i, (sx, sy) in enumerate(seeds):
        for dx, dy in offsets:
            ext_pts.append([sx+dx, sy+dy]); ext_ids.append(i)
    ext_pts = np.array(ext_pts, dtype=np.float32)
    ext_ids = np.array(ext_ids, dtype=np.int32)

    ys, xs  = np.mgrid[0:H, 0:W]
    if warp is not None:
        qx = np.clip(xs.astype(np.float32) + warp[0], 0, W-1)
        qy = np.clip(ys.astype(np.float32) + warp[1], 0, H-1)
    else:
        qx = xs.astype(np.float32)
        qy = ys.astype(np.float32)

    pix_pts = np.column_stack([qx.ravel(), qy.ravel()])
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
    else:
        return (np.floor(((xn - yn) % 1.0) * n).astype(np.int32) % 2) == 0


# ── Outline drawing ───────────────────────────────────────────────────────────

def _draw_outlines(canvas: np.ndarray, zone_map: np.ndarray,
                   outline_width: int,
                   outline_color: tuple = (0, 0, 0)) -> None:
    if outline_width <= 0:
        return
    diff_r = (zone_map[:, :-1] != zone_map[:, 1:])
    diff_d = (zone_map[:-1, :] != zone_map[1:, :])
    border = np.zeros(zone_map.shape, dtype=np.uint8)
    border[:, :-1][diff_r] = 255
    border[:-1, :][diff_d] = 255
    if outline_width > 1:
        border = cv2.dilate(border, np.ones((outline_width, outline_width), np.uint8))
    b, g, r = int(outline_color[0]), int(outline_color[1]), int(outline_color[2])
    canvas[border > 0, 0] = b
    canvas[border > 0, 1] = g
    canvas[border > 0, 2] = r
    if canvas.shape[2] == 4:
        canvas[border > 0, 3] = 255


# ── Generator ─────────────────────────────────────────────────────────────────

class DazzleGenerator(BaseGenerator):
    name = "Dazzle"
    description = (
        "Toroidal Voronoi zones each with independent stripe direction/density. "
        "Organic warp (warp_strength > 0) gives giraffe-patch zone shapes. "
        "outline_width draws thick borders between zones."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["dazzle"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        num_zones      = max(1,  int(params.get("num_zones",       12)))
        stripe_density = max(0,  int(params.get("stripe_density",   8)))
        density_var    = float(params.get("density_variation",   0.5))
        directions     = params.get("directions",               "mixed")
        outline_width  = int(params.get("outline_width",          2))
        high_contrast  = bool(params.get("high_contrast",        True))
        warp_strength  = float(params.get("warp_strength",       0.0))
        warp_scale     = float(params.get("warp_scale",          0.3))
        transparent    = bool(params.get("transparent_bg",       False))
        seed           = int(params.get("seed",                  42))

        rng = np.random.default_rng(seed)
        n   = max(1, len(colors))
        bg_idx, exclude = get_bg_params(params, n)
        fg_colors = make_fg_colors(colors, bg_idx, exclude)
        nf = max(1, len(fg_colors))
        bg_color = colors[bg_idx]
        W, H = width, height

        warp = _build_warp(H, W, warp_strength, warp_scale, seed)

        seeds    = np.column_stack([rng.uniform(0,W,num_zones),
                                    rng.uniform(0,H,num_zones)])
        zone_map = _toroidal_voronoi(seeds, W, H, warp)

        dir_pool = {"mixed": _ALL_DIRS, "striped": ["vertical","horizontal"],
                    "diagonal": ["diagonal+","diagonal-"]}.get(directions, _ALL_DIRS)
        zone_dirs = [dir_pool[int(rng.integers(0, len(dir_pool)))]
                     for _ in range(num_zones)]
        zone_ns   = [max(0, stripe_density +
                         int(rng.integers(-max(0,int(density_var*stripe_density)),
                                          max(0,int(density_var*stripe_density))+1)))
                     for _ in range(num_zones)]

        zone_ca, zone_cb = [], []
        for z in range(num_zones):
            if transparent:
                ci = (z * 2 if high_contrast else z) % nf
                zone_ca.append(fg_colors[ci])
                zone_cb.append(bg_color)
            else:
                if high_contrast and nf >= 2:
                    ci0, ci1 = (z*2)%nf, (z*2+1)%nf
                else:
                    ci0, ci1 = z%nf, (z+1)%nf
                zone_ca.append(fg_colors[ci0])
                zone_cb.append(fg_colors[ci1])

        ys_grid, xs_grid = np.mgrid[0:H, 0:W]
        xn = xs_grid.astype(np.float32) / W
        yn = ys_grid.astype(np.float32) / H

        # Render BGR first; transparency applied after
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for z in range(num_zones):
            zmask = (zone_map == z)
            if not zmask.any(): continue
            ab = _stripe_binary(xn, yn, zone_dirs[z], zone_ns[z])
            r0,g0,b0 = zone_ca[z]; r1,g1,b1 = zone_cb[z]
            canvas[zmask & ab]  = (int(b0), int(g0), int(r0))
            canvas[zmask & ~ab] = (int(b1), int(g1), int(r1))

        # Draw outlines on BGR then optionally convert to BGRA
        if outline_width > 0:
            r0, g0, b0 = colors[bg_idx]
            outline_bgr = (int(b0), int(g0), int(r0))
            canvas4 = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
            _draw_outlines(canvas4, zone_map, outline_width, outline_bgr)
            canvas  = cv2.cvtColor(canvas4, cv2.COLOR_BGRA2BGR)

        if transparent:
            return apply_transparent_bg(canvas, colors, bg_idx)
        return canvas
