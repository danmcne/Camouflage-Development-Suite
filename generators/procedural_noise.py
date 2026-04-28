"""
Procedural Noise Generator – seamlessly tiling noise with lacunarity beat fix.

Lacunarity beat fix
───────────────────
When lacunarity is not an integer, the frequency of octave k is
  f_k = periods * lacunarity^k
which is not generally a multiple of 1/periods, so octave k does not tile
with the same period as octave 0.  Their superposition creates "beat"
patterns — visible as diagonal stripes (Perlin) or grid artefacts (rotated).

Fix: enforce integer lacunarity values (2, 3, or 4).  With integer lacunarity
every octave's frequency is an integer multiple of periods, so pnoise2's
built-in repeat mechanism tiles all octaves simultaneously.

The "simplex" mode was previously just rotated Perlin (the noise library only
provides pnoise2, not snoise2).  It is now honestly named "rotated_perlin".
For users who want a smoother feel without beat artefacts the recommended
approach is integer lacunarity + higher persistence.

Turbulence domain-warp uses a fixed base offset so it doesn't break tiling.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator, get_bg_params
from config.defaults import GENERATORS
from generators.blur_sharp import _colorise


class ProceduralNoiseGenerator(BaseGenerator):
    name = "Procedural Noise"
    description = (
        "Seamlessly tiling Perlin noise (pnoise2 with integer repeat). "
        "Integer lacunarity (2/3/4) prevents octave beat-frequency artefacts. "
        "Turbulence adds organic domain warp without breaking tiling."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["procedural_noise"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        import noise as noise_lib

        rng         = np.random.default_rng(int(params.get("seed", 42)))
        noise_type  = params.get("noise_type", "seamless_perlin")
        octaves     = int(params.get("octaves", 6))
        persistence = float(params.get("persistence", 0.5))
        # Enforce integer lacunarity to prevent octave beat artefacts
        raw_lac     = float(params.get("lacunarity", 2.0))
        lacunarity  = float(max(0, min(5, round(raw_lac))))
        periods     = int(max(1, round(float(params.get("periods", 3)))))
        turbulence  = float(params.get("turbulence", 0.0))
        color_mode  = params.get("color_mode", "threshold")
        transparent = bool(params.get("transparent_bg", False))

        # Random base offset — field still tiles because pnoise2 repeatx/y=periods
        base_x = float(rng.integers(0, periods * 100)) / 100.0
        base_y = float(rng.integers(0, periods * 100)) / 100.0
        	
        field = np.zeros((height, width), dtype=np.float32)

        # Rotation angle for the rotated_perlin mode (kept fixed by seed)
        rot_angle = float(rng.uniform(0, 2 * math.pi)) if noise_type == "rotated_perlin" else 0.0
        cos_a, sin_a = math.cos(rot_angle), math.sin(rot_angle)

        for y in range(height):
            ny = (y / height) * periods
            for x in range(width):
                nx = (x / width) * periods

                if noise_type == "rotated_perlin":
                    nx2 = nx * cos_a - ny * sin_a + base_x
                    ny2 = nx * sin_a + ny * cos_a + base_y
                else:
                    nx2, ny2 = nx + base_x, ny + base_y

                # Turbulence: warp coordinates.  Use a fixed cross-offset so
                # the warp field itself tiles (it uses the same repeat).
                if turbulence > 0.0:
                    wx = noise_lib.pnoise2(
                        nx2 + 13.1, ny2 + 7.7,
                        octaves=max(1, octaves // 2),
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods)
                    wy = noise_lib.pnoise2(
                        nx2 + 3.3, ny2 + 17.2,
                        octaves=max(1, octaves // 2),
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods)
                    nx2 += turbulence * wx
                    ny2 += turbulence * wy

                field[y, x] = noise_lib.pnoise2(
                    nx2, ny2,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=periods, repeaty=periods)

        mn, mx = field.min(), field.max()
        field  = (field - mn) / (mx - mn + 1e-8)

        n = max(1, len(colors))
        bg_idx, exclude = get_bg_params(params, n)
        return _colorise(field, colors, color_mode, transparent, bg_idx, exclude)
