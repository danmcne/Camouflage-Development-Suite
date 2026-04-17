"""
Urban Geometric Generator – seamless geometric primitives.

Three placement modes
─────────────────────
  tiled       Original grid/offset tiling with per-cell jitter.
              Seams handled by 9-copy toroidal drawing.

  random      Scatter N shapes at random positions, any size within
              [scale_min, scale_max] fraction of canvas, any rotation.
              Toroidal via 9-copy draw.

  field_driven  Like 'random' but positions are importance-sampled
              from a Perlin density field, and rotation follows a
              Perlin orientation field.  Produces the same correlated-
              placement effect as field-driven Collage.

All modes draw via the 9-copy toroidal stamp so shapes wrapping the right
or bottom edge appear at the left or top edge respectively.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS


# ── Field helpers (shared logic with collage) ─────────────────────────────────

def _build_fields(width, height, field_scale, rng):
    """Return (density, orient) float32 [0,1] fields, shape (H, W)."""
    fw = max(16, int(width  * field_scale))
    fh = max(16, int(height * field_scale))

    try:
        import noise as noise_lib
        base_x = float(rng.uniform(0, 100))
        base_y = float(rng.uniform(0, 100))
        periods = 3
        den_raw = np.zeros((fh, fw), dtype=np.float32)
        ori_raw = np.zeros((fh, fw), dtype=np.float32)
        for fy in range(fh):
            ny = (fy / fh) * periods
            for fx in range(fw):
                nx = (fx / fw) * periods
                den_raw[fy, fx] = noise_lib.pnoise2(
                    nx + base_x, ny + base_y,
                    octaves=4, persistence=0.5, lacunarity=2.0,
                    repeatx=periods, repeaty=periods)
                ori_raw[fy, fx] = noise_lib.pnoise2(
                    nx + base_x + 50, ny + base_y + 37,
                    octaves=3, persistence=0.5, lacunarity=2.0,
                    repeatx=periods, repeaty=periods)
    except ImportError:
        den_raw = rng.random((fh, fw)).astype(np.float32)
        ori_raw = rng.random((fh, fw)).astype(np.float32)
        k = max(3, (min(fh, fw) // 3) | 1)
        den_raw = cv2.GaussianBlur(den_raw, (k, k), k / 6)
        ori_raw = cv2.GaussianBlur(ori_raw, (k, k), k / 6)

    def _norm(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-8)

    den = cv2.resize(_norm(den_raw), (width, height), interpolation=cv2.INTER_LINEAR)
    ori = cv2.resize(_norm(ori_raw), (width, height), interpolation=cv2.INTER_LINEAR)
    return den, ori


def _importance_sample(density, density_contrast, count, rng):
    """Importance-sample (x,y) positions from density field."""
    h, w = density.shape
    power   = 1.0 + density_contrast * 8.0
    weights = density.ravel() ** power
    total   = weights.sum()
    if total <= 0:
        weights = np.ones_like(weights)
        total   = weights.sum()
    probs = weights / total
    flat  = rng.choice(len(probs), size=count, replace=True, p=probs)
    ys, xs = np.unravel_index(flat, (h, w))
    return list(zip(xs.tolist(), ys.tolist()))


# ── Primitive polygon builders ────────────────────────────────────────────────

def _make_primitive(primitive: str, cx: int, cy: int,
                    size: int, angle_deg: float) -> np.ndarray:
    """
    Return (N,1,2) int32 polygon for one primitive centred at (cx,cy).
    `angle_deg` rotates the shape.
    """
    r   = size // 2
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    def _rot(px, py):
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        return int(cx + rx), int(cy + ry)

    if primitive == "hexagon":
        pts = [_rot(r * math.cos(math.radians(60*i)),
                    r * math.sin(math.radians(60*i))) for i in range(6)]
    elif primitive == "triangle":
        h3 = r * math.sqrt(3) / 2
        pts = [_rot(0, -r), _rot(h3, r//2), _rot(-h3, r//2)]
    elif primitive == "diamond":
        pts = [_rot(0,-r), _rot(r,0), _rot(0,r), _rot(-r,0)]
    else:   # offset_grid → square
        pts = [_rot(-r,-r), _rot(r,-r), _rot(r,r), _rot(-r,r)]

    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


# ── Generator ─────────────────────────────────────────────────────────────────

class UrbanGeometricGenerator(BaseGenerator):
    name = "Urban Geometric"
    description = (
        "Tiled geometric primitives (hexagon, triangle, diamond, square). "
        "Three placement modes: tiled grid with jitter (original), random scatter, "
        "or field-driven clustered placement like field-collage."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["urban_geometric"]

    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],
        params: dict,
    ) -> np.ndarray:
        primitive      = params.get("primitive",      "hexagon")
        cell_size      = int(params.get("cell_size",  40))
        size_count     = int(params.get("size_count", 2))
        size_ratio     = float(params.get("size_ratio", 0.5))
        jitter         = float(params.get("jitter",   0.12))
        outline_w      = int(params.get("outline_width", 0))
        placement_mode = params.get("placement_mode", "tiled")
        num_shapes     = int(params.get("num_shapes", 60))
        scale_min      = float(params.get("scale_min", 0.05))
        scale_max      = float(params.get("scale_max", 0.20))
        rot_range      = float(params.get("rotation_range", 180))
        field_scale    = float(params.get("field_scale",    0.25))
        density_contrast=float(params.get("density_contrast",0.7))
        orient_coh     = float(params.get("orient_coherence",0.6))
        transparent    = bool(params.get("transparent_bg", False))
        seed           = int(params.get("seed",       42))

        rng = np.random.default_rng(seed)
        n   = max(1, len(colors))
        W, H = width, height

        channels = 4 if transparent else 3
        canvas   = np.zeros((H, W, channels), dtype=np.uint8)
        if not transparent and colors:
            r0, g0, b0 = colors[0]
            canvas[:] = (int(b0), int(g0), int(r0))

        # 9-copy toroidal offsets
        toro = [(-W,-H),(0,-H),(W,-H),
                (-W, 0),(0, 0),(W, 0),
                (-W, H),(0, H),(W, H)]

        def _draw_tiles(tile_list):
            """tile_list: [(pts_np_int32, color_bgr), ...]"""
            for pts_base, c_bgr in tile_list:
                for dx, dy in toro:
                    shifted = pts_base + np.array([[[dx, dy]]], dtype=np.int32)
                    if transparent:
                        cv2.fillPoly(canvas,  [shifted], c_bgr + (255,))
                        if outline_w:
                            cv2.polylines(canvas, [shifted], True,
                                          (0,0,0,255), outline_w)
                    else:
                        cv2.fillPoly(canvas,  [shifted], c_bgr)
                        if outline_w:
                            cv2.polylines(canvas, [shifted], True,
                                          (0,0,0), outline_w)

        # ── Placement modes ───────────────────────────────────────────────────

        if placement_mode == "tiled":
            sizes = [cell_size]
            for _ in range(1, size_count):
                sizes.append(max(4, int(sizes[-1] * size_ratio)))

            all_tiles = []
            for sz in sizes:
                raw = self._collect_tiles(primitive, sz, jitter, W, H, rng)
                for pts, _ in raw:
                    ci = int(rng.integers(0, n))
                    r, g, b = colors[ci]
                    all_tiles.append((pts, (int(b), int(g), int(r))))

            order = rng.permutation(len(all_tiles))
            _draw_tiles([all_tiles[i] for i in order])

        elif placement_mode == "random":
            base = min(W, H)
            tiles = []
            for _ in range(num_shapes):
                cx    = int(rng.integers(0, W))
                cy    = int(rng.integers(0, H))
                sz    = max(4, int(rng.uniform(scale_min, scale_max) * base))
                angle = float(rng.uniform(-rot_range, rot_range))
                ci    = int(rng.integers(0, n))
                r, g, b = colors[ci]
                pts   = _make_primitive(primitive, cx, cy, sz, angle)
                tiles.append((pts, (int(b), int(g), int(r))))
            _draw_tiles(tiles)

        else:   # field_driven
            density_f, orient_f = _build_fields(W, H, field_scale, rng)
            positions = _importance_sample(density_f, density_contrast,
                                           num_shapes, rng)
            base = min(W, H)
            tiles = []
            for cx, cy in positions:
                sz  = max(4, int(rng.uniform(scale_min, scale_max) * base))
                field_angle = orient_f[cy, cx] * 360.0 - 180.0
                rand_angle  = float(rng.uniform(-rot_range, rot_range))
                angle = (field_angle * orient_coh
                         + rand_angle * (1.0 - orient_coh))
                ci    = int(rng.integers(0, n))
                r, g, b = colors[ci]
                pts   = _make_primitive(primitive, cx, cy, sz, angle)
                tiles.append((pts, (int(b), int(g), int(r))))
            _draw_tiles(tiles)

        return canvas

    # ── tiled grid collectors (unchanged from original) ───────────────────────

    def _collect_tiles(self, primitive, cell, jitter, W, H, rng):
        if primitive == "hexagon":    return self._hex_tiles(cell, jitter, W, H, rng)
        elif primitive == "triangle": return self._tri_tiles(cell, jitter, W, H, rng)
        elif primitive == "diamond":  return self._diamond_tiles(cell, jitter, W, H, rng)
        else:                         return self._grid_tiles(cell, jitter, W, H, rng)

    def _hex_tiles(self, cell, jitter, W, H, rng):
        tiles = []
        r     = cell
        hex_w = int(math.sqrt(3) * r)
        hex_h = int(1.5 * r)
        j     = max(1, int(r * jitter))
        for row in range(-1, H // max(hex_h, 1) + 2):
            for col in range(-1, W // max(hex_w, 1) + 2):
                cx = col * hex_w + (hex_w // 2 if row % 2 else 0)
                cy = row * hex_h
                cx += int(rng.integers(-j, j + 1))
                cy += int(rng.integers(-j, j + 1))
                tiles.append((self._hex_pts(cx, cy, r), None))
        return tiles

    def _hex_pts(self, cx, cy, r):
        pts = [[int(cx + r * math.cos(math.radians(60*i))),
                int(cy + r * math.sin(math.radians(60*i)))]
               for i in range(6)]
        return np.array(pts, np.int32).reshape(-1, 1, 2)

    def _tri_tiles(self, cell, jitter, W, H, rng):
        tiles = []
        j = max(1, int(cell * jitter))
        for row in range(-1, H // max(cell, 1) + 2):
            for col in range(-1, W // max(cell, 1) + 2):
                x0, y0 = col * cell, row * cell
                for up in (True, False):
                    if up:
                        pts = np.array([
                            [x0+rng.integers(-j,j+1), y0+cell+rng.integers(-j,j+1)],
                            [x0+cell//2+rng.integers(-j,j+1), y0+rng.integers(-j,j+1)],
                            [x0+cell+rng.integers(-j,j+1), y0+cell+rng.integers(-j,j+1)],
                        ], np.int32).reshape(-1, 1, 2)
                    else:
                        pts = np.array([
                            [x0+rng.integers(-j,j+1), y0+rng.integers(-j,j+1)],
                            [x0+cell+rng.integers(-j,j+1), y0+rng.integers(-j,j+1)],
                            [x0+cell//2+rng.integers(-j,j+1), y0+cell+rng.integers(-j,j+1)],
                        ], np.int32).reshape(-1, 1, 2)
                    tiles.append((pts, None))
        return tiles

    def _diamond_tiles(self, cell, jitter, W, H, rng):
        tiles = []
        j  = max(1, int(cell * jitter))
        hw = cell // 2
        for row in range(-1, H // max(cell, 1) + 2):
            for col in range(-1, W // max(cell, 1) + 2):
                cx = col * cell + (hw if row % 2 else 0)
                cy = row * cell
                pts = np.array([
                    [cx, cy-hw+rng.integers(-j,j+1)],
                    [cx+hw+rng.integers(-j,j+1), cy],
                    [cx, cy+hw+rng.integers(-j,j+1)],
                    [cx-hw+rng.integers(-j,j+1), cy],
                ], np.int32).reshape(-1, 1, 2)
                tiles.append((pts, None))
        return tiles

    def _grid_tiles(self, cell, jitter, W, H, rng):
        tiles = []
        j = max(1, int(cell * jitter))
        for row in range(-1, H // max(cell, 1) + 2):
            for col in range(-1, W // max(cell, 1) + 2):
                off = cell // 2 if row % 2 else 0
                x0  = col * cell + off + rng.integers(-j, j + 1)
                y0  = row * cell + rng.integers(-j, j + 1)
                pts = np.array(
                    [[x0,y0],[x0+cell,y0],[x0+cell,y0+cell],[x0,y0+cell]],
                    np.int32).reshape(-1, 1, 2)
                tiles.append((pts, None))
        return tiles
