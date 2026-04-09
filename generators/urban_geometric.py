"""
Urban Geometric Generator – seamless via 9-copy toroidal drawing.

Seam fix: after collecting all tile polygon coordinates, each tile is drawn
not just at its natural position but also at offsets (±W, 0), (0, ±H), and
(±W, ±H).  OpenCV's fillPoly clips to canvas bounds automatically, so tiles
that straddle the right or bottom edge also paint the correct pixels at the
left and top edges — exactly what is needed for seamless tiling.

This approach does NOT require the grid period to divide the canvas size,
which would distort the tile shapes.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS


class UrbanGeometricGenerator(BaseGenerator):
    name = "Urban Geometric"
    description = (
        "Tiled geometric primitives with multiple cell sizes. "
        "Perfectly seamless at any canvas size via toroidal tile drawing."
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
        primitive   = params.get("primitive", "hexagon")
        cell_size   = int(params.get("cell_size", 40))
        size_count  = int(params.get("size_count", 2))
        size_ratio  = float(params.get("size_ratio", 0.5))
        jitter      = float(params.get("jitter", 0.12))
        outline_w   = int(params.get("outline_width", 0))
        transparent = bool(params.get("transparent_bg", False))
        seed        = int(params.get("seed", 42))

        rng = np.random.default_rng(seed)
        n   = max(1, len(colors))

        sizes = [cell_size]
        for _ in range(1, size_count):
            sizes.append(max(4, int(sizes[-1] * size_ratio)))

        channels = 4 if transparent else 3
        canvas   = np.zeros((height, width, channels), dtype=np.uint8)
        if not transparent and colors:
            r0, g0, b0 = colors[0]
            canvas[:] = (int(b0), int(g0), int(r0))

        # Collect all tiles as (pts, color_bgr)
        all_tiles = []
        for sz in sizes:
            raw = self._collect_tiles(primitive, sz, jitter, width, height, rng)
            for pts, _ in raw:
                ci = int(rng.integers(0, n))
                r, g, b = colors[ci]
                all_tiles.append((pts, (int(b), int(g), int(r))))

        # Shuffle so sizes interleave
        order = rng.permutation(len(all_tiles))

        # Draw each tile at 9 toroidal positions
        offsets = [(-width, -height), (0, -height), (width, -height),
                   (-width,       0), (0,       0), (width,       0),
                   (-width,  height), (0,  height), (width,  height)]

        for idx in order:
            pts_base, color_bgr = all_tiles[idx]
            for dx, dy in offsets:
                shifted = pts_base + np.array([[[dx, dy]]], dtype=np.int32)
                if transparent:
                    cv2.fillPoly(canvas, [shifted], color_bgr + (255,))
                    if outline_w:
                        cv2.polylines(canvas, [shifted], True, (0, 0, 0, 255), outline_w)
                else:
                    cv2.fillPoly(canvas, [shifted], color_bgr)
                    if outline_w:
                        cv2.polylines(canvas, [shifted], True, (0, 0, 0), outline_w)

        return canvas

    # ── tile collectors ───────────────────────────────────────────────────────

    def _collect_tiles(self, primitive, cell, jitter, W, H, rng):
        if primitive == "hexagon":
            return self._hex_tiles(cell, jitter, W, H, rng)
        elif primitive == "triangle":
            return self._tri_tiles(cell, jitter, W, H, rng)
        elif primitive == "diamond":
            return self._diamond_tiles(cell, jitter, W, H, rng)
        else:
            return self._grid_tiles(cell, jitter, W, H, rng)

    def _hex_tiles(self, cell, jitter, W, H, rng):
        tiles  = []
        r      = cell
        hex_w  = int(math.sqrt(3) * r)
        hex_h  = int(1.5 * r)
        j      = max(1, int(r * jitter))
        # +1 extra row/col: the 9-copy drawing handles seams, not the extension
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
