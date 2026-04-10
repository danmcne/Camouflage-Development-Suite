"""
Procedural Noise Generator – seamless pnoise2(repeatx=N) with optional transparent background.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS
from generators.blur_sharp import _colorise   # shared helper


class ProceduralNoiseGenerator(BaseGenerator):
    name = "Procedural Noise"
    description = (
        "Seamlessly tiling Perlin noise (pnoise2 repeat). "
        "'Tile periods' controls blob size. Both modes are fully seamless."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["procedural_noise"]

    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],
        params: dict,
        ) -> np.ndarray:
        import noise as noise_lib
        import numpy as np
        import math
    
        rng         = np.random.default_rng(int(params.get("seed", 42)))
        noise_type  = params.get("noise_type", "seamless_perlin")
        octaves     = int(params.get("octaves", 6))
        persistence = float(params.get("persistence", 0.5))
        lacunarity  = float(params.get("lacunarity", 2.0))
        periods     = float(params.get("periods", 3))
        turbulence  = float(params.get("turbulence", 0.0))
        color_mode  = params.get("color_mode", "threshold")
        transparent = bool(params.get("transparent_bg", False))
    
        # Continuous offsets → seed actually matters, still tiles
        base_x = rng.uniform(0.0, periods)
        base_y = rng.uniform(0.0, periods)
    
        # Optional rotation to reduce axis-aligned artifacts
        angle = rng.uniform(0, 2 * math.pi)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
    
        field = np.zeros((height, width), dtype=np.float32)
    
        for y in range(height):
            ny = (y / height) * periods
            for x in range(width):
                nx = (x / width) * periods
    
                # Rotate domain (breaks grid feel)
                nx_r = nx * cos_a - ny * sin_a + base_x
                ny_r = nx * sin_a + ny * cos_a + base_y
    
                # Optional turbulence (domain warping)
                if turbulence > 0.0:
                    wx = noise_lib.pnoise2(
                        nx_r + 13.1, ny_r + 7.7,
                        octaves=max(1, octaves // 2),
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods
                    )
                    wy = noise_lib.pnoise2(
                        nx_r + 3.3, ny_r + 17.2,
                        octaves=max(1, octaves // 2),
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods
                    )
                    nx_r += turbulence * wx
                    ny_r += turbulence * wy
    
                if noise_type == "seamless_simplex":
                    angle = math.pi / 4
                    nx2   = nx_r * math.cos(angle) - ny_r * math.sin(angle)
                    ny2   = nx_r * math.sin(angle) + ny_r * math.cos(angle)
                    v1 = noise_lib.pnoise2(nx_r, ny_r,
                                       octaves=octaves, persistence=persistence,
                                       lacunarity=lacunarity,
                                       repeatx=periods, repeaty=periods)
                    v2 = noise_lib.pnoise2(nx2 + base_x * 0.3, ny2 + base_y * 0.3,
                                       octaves=octaves, persistence=persistence,
                                       lacunarity=lacunarity,
                                       repeatx=periods, repeaty=periods)
                    field[y, x] = (v1 + v2) * 0.5

                else: 
                    v = noise_lib.pnoise2(
                        nx_r, ny_r,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods
                    )
    
                    field[y, x] = v
    
        # Normalize to [0,1]
        mn, mx = field.min(), field.max()
        if mx > mn:
            field = (field - mn) / (mx - mn)
        else:
            field[:] = 0.5
    
        return _colorise(field, colors, color_mode, transparent)
    
