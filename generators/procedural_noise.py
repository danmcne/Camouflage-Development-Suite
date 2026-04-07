"""
Procedural Noise Generator – guaranteed seamless tiling via pnoise2 repeat parameter.

The `noise.pnoise2(x, y, repeatx=R, repeaty=R)` call guarantees that the output
tiles perfectly at period R in noise-space coordinates.  We map the canvas so
exactly `periods` full noise cycles fit across each axis, giving a seamless tile.

Both "seamless_perlin" and "seamless_simplex" use this same mechanism;
seamless_simplex adds a second octave pass with a rotated coordinate frame
for a slightly smoother, less grid-aligned appearance.

The old snoise4 / 4D-torus approach has been removed — it caused segfaults
on many builds of the noise library.
"""
from __future__ import annotations
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS


class ProceduralNoiseGenerator(BaseGenerator):
    name = "Procedural Noise"
    description = (
        "Seamlessly tiling Perlin noise. "
        "'Tile periods' controls blob size (low = large, high = fine). "
        "Both modes are fully seamless."
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

        rng         = np.random.default_rng(int(params.get("seed", 42)))
        noise_type  = params.get("noise_type", "seamless_perlin")
        octaves     = int(params.get("octaves", 6))
        persistence = float(params.get("persistence", 0.5))
        lacunarity  = float(params.get("lacunarity", 2.0))
        periods     = int(params.get("periods", 3))
        turbulence  = float(params.get("turbulence", 0.0))
        color_mode  = params.get("color_mode", "threshold")

        # Integer base offsets keep the repeat-based tiling valid
        base_x = int(rng.integers(0, 500)) * periods
        base_y = int(rng.integers(0, 500)) * periods

        field = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            # Noise coords: map [0,height) → [0, periods)
            ny = (y / height) * periods + base_y
            for x in range(width):
                nx = (x / width) * periods + base_x

                if turbulence > 0.0:
                    wx = noise_lib.pnoise2(
                        nx + 1.7, ny + 9.2,
                        octaves=max(1, octaves // 2),
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods,
                    )
                    wy = noise_lib.pnoise2(
                        nx + 8.3, ny + 2.8,
                        octaves=max(1, octaves // 2),
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods,
                    )
                    nx += turbulence * wx
                    ny_w = ny + turbulence * wy
                else:
                    ny_w = ny

                if noise_type == "seamless_simplex":
                    # Two pnoise2 passes at different angles → less grid-aligned
                    import math
                    angle = math.pi / 4
                    nx2 = nx * math.cos(angle) - ny_w * math.sin(angle)
                    ny2 = nx * math.sin(angle) + ny_w * math.cos(angle)
                    v1 = noise_lib.pnoise2(
                        nx, ny_w,
                        octaves=octaves, persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods,
                    )
                    v2 = noise_lib.pnoise2(
                        nx2 + base_x * 0.3, ny2 + base_y * 0.3,
                        octaves=octaves, persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods,
                    )
                    field[y, x] = (v1 + v2) * 0.5
                else:
                    field[y, x] = noise_lib.pnoise2(
                        nx, ny_w,
                        octaves=octaves, persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods,
                    )

        # Normalise to [0, 1]
        fmin, fmax = field.min(), field.max()
        if fmax > fmin:
            field = (field - fmin) / (fmax - fmin)
        else:
            field[:] = 0.5

        n = max(1, len(colors))
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        if color_mode == "voronoi":
            seeds  = np.linspace(0.05, 0.95, n)
            exp_f  = field[:, :, np.newaxis]
            exp_s  = seeds[np.newaxis, np.newaxis, :]
            nearest= np.argmin(np.abs(exp_f - exp_s), axis=2)
            for i, (r, g, b) in enumerate(colors):
                canvas[nearest == i] = (int(b), int(g), int(r))
        else:
            thresholds = np.linspace(0.0, 1.0 + 1e-6, n + 1)
            for i, (r, g, b) in enumerate(colors):
                mask = (field >= thresholds[i]) & (field < thresholds[i + 1])
                canvas[mask] = (int(b), int(g), int(r))

        return canvas
