"""
Urban Geometric Generator – tiled primitives with multi-size and optional transparent background.

When `transparent_bg` is True the canvas is initialised fully transparent (BGRA)
and only the tile interiors are drawn opaque.  Use this in the second generator
layer so the space between tiles doesn't bleed through.
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
        "Tiled geometric primitives with multiple cell sizes for multi-scale patterns. "
        "Enable 'Transparent background' when using as a second layer."
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
        primitive    = params.get("primitive", "hexagon")
        cell_size    = int(params.get("cell_size", 40))
        size_count   = int(params.get("size_count", 2))
        size_ratio   = float(params.get("size_ratio", 0.5))
        jitter       = float(params.get("jitter", 0.12))
        outline_w    = int(params.get("outline_width", 0))
        transparent  = bool(params.get("transparent_bg", False))
        seed         = int(params.get("seed", 42))

        rng = np.random.default_rng(seed)
        n   = max(1, len(colors))

        # Build size list large → small
        sizes = [cell_size]
        for _ in range(1, size_count):
            sizes.append(max(4, int(sizes[-1] * size_ratio)))

        if transparent:
            # 4-channel BGRA; bg fully transparent
            canvas = np.zeros((height, width, 4), dtype=np.uint8)
        else:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            if colors:
                r0, g0, b0 = colors[0]
                canvas[:] = (int(b0), int(g0), int(r0))

        # Collect tiles from all sizes
        all_tiles = []
        for sz in sizes:
            tiles = self._collect_tiles(primitive, sz, jitter, width, height, rng)
            for pts, _ in tiles:
                ci = int(rng.integers(0, n))
                all_tiles.append((pts, colors[ci]))

        # Shuffle draw order so sizes interleave
        order = rng.permutation(len(all_tiles))

        for idx in order:
            pts, (r, g, b) = all_tiles[idx]
            if transparent:
                color = (int(b), int(g), int(r), 255)
                cv2.fillPoly(canvas, [pts], color)
                if outline_w:
                    cv2.polylines(canvas, [pts], True, (0, 0, 0, 255), outline_w)
            else:
                color = (int(b), int(g), int(r))
                cv2.fillPoly(canvas, [pts], color)
                if outline_w:
                    cv2.polylines(canvas, [pts], True, (0, 0, 0), outline_w)

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
        tiles = []
        r     = cell
        hex_w = int(math.sqrt(3) * r)
        hex_h = int(1.5 * r)
        j     = max(1, int(r * jitter))
        for row in range(-2, H // max(hex_h, 1) + 3):
            for col in range(-2, W // max(hex_w, 1) + 3):
                cx = col * hex_w + (hex_w // 2 if row % 2 else 0)
                cy = row * hex_h
                cx += int(rng.integers(-j, j + 1))
                cy += int(rng.integers(-j, j + 1))
                tiles.append((self._hex_pts(cx, cy, r), None))
        return tiles

    def _hex_pts(self, cx, cy, r):
        pts = []
        for i in range(6):
            a = math.radians(60 * i)
            pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
        return np.array(pts, np.int32).reshape(-1, 1, 2)

    def _tri_tiles(self, cell, jitter, W, H, rng):
        tiles = []
        j = max(1, int(cell * jitter))
        for row in range(-2, H // max(cell, 1) + 3):
            for col in range(-2, W // max(cell, 1) + 3):
                x0, y0 = col * cell, row * cell
                for up in (True, False):
                    if up:
                        pts = np.array([
                            [x0 + rng.integers(-j,j+1), y0+cell+rng.integers(-j,j+1)],
                            [x0+cell//2+rng.integers(-j,j+1), y0+rng.integers(-j,j+1)],
                            [x0+cell+rng.integers(-j,j+1), y0+cell+rng.integers(-j,j+1)],
                        ], np.int32).reshape(-1,1,2)
                    else:
                        pts = np.array([
                            [x0+rng.integers(-j,j+1), y0+rng.integers(-j,j+1)],
                            [x0+cell+rng.integers(-j,j+1), y0+rng.integers(-j,j+1)],
                            [x0+cell//2+rng.integers(-j,j+1), y0+cell+rng.integers(-j,j+1)],
                        ], np.int32).reshape(-1,1,2)
                    tiles.append((pts, None))
        return tiles

    def _diamond_tiles(self, cell, jitter, W, H, rng):
        tiles = []
        j  = max(1, int(cell * jitter))
        hw = cell // 2
        for row in range(-2, H // max(cell, 1) + 3):
            for col in range(-2, W // max(cell, 1) + 3):
                cx = col * cell + (hw if row % 2 else 0)
                cy = row * cell
                pts = np.array([
                    [cx, cy-hw+rng.integers(-j,j+1)],
                    [cx+hw+rng.integers(-j,j+1), cy],
                    [cx, cy+hw+rng.integers(-j,j+1)],
                    [cx-hw+rng.integers(-j,j+1), cy],
                ], np.int32).reshape(-1,1,2)
                tiles.append((pts, None))
        return tiles

    def _grid_tiles(self, cell, jitter, W, H, rng):
        tiles = []
        j = max(1, int(cell * jitter))
        for row in range(-2, H // max(cell, 1) + 3):
            for col in range(-2, W // max(cell, 1) + 3):
                off = cell // 2 if row % 2 else 0
                x0  = col * cell + off + rng.integers(-j, j+1)
                y0  = row * cell + rng.integers(-j, j+1)
                pts = np.array([
                    [x0,y0],[x0+cell,y0],[x0+cell,y0+cell],[x0,y0+cell],
                ], np.int32).reshape(-1,1,2)
                tiles.append((pts, None))
        return tiles
