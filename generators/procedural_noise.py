"""
Procedural Noise Generator – seamlessly tiling noise with two modes.

seamless_perlin
───────────────
Standard pnoise2 with integer repeat.  Integer lacunarity (2, 3, 4) prevents
octave beat-frequency artefacts (visible as diagonal stripes / grid lines when
lacunarity is non-integer).

seamless_simplex
────────────────
True seamless simplex noise via the 4D torus trick:
  Map each pixel (x, y) to a 4D point on a hyper-torus:
    (cos(2πx/W)·R,  sin(2πx/W)·R,  cos(2πy/H)·R,  sin(2πy/H)·R)
  Then evaluate opensimplex.noise4().  Because the mapping is periodic in
  both x and y, the resulting field tiles perfectly with zero seam and
  exhibits no linear artefacts (simplex has no axis-aligned bias).
  Uses the `opensimplex` library; falls back to seamless_perlin if absent.

Turbulence domain-warp uses a fixed cross-offset so it doesn't break tiling.
"""
from __future__ import annotations
import math
import numpy as np
from generators.base import BaseGenerator, get_bg_params
from config.defaults import GENERATORS
from generators.blur_sharp import _colorise


class ProceduralNoiseGenerator(BaseGenerator):
    name = "Procedural Noise"
    description = (
        "Seamlessly tiling noise. "
        "seamless_perlin uses pnoise2 with integer repeat and lacunarity. "
        "seamless_simplex uses the 4D torus trick — perfectly seamless, "
        "no linear artefacts, no axis-aligned bias."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["procedural_noise"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        rng         = np.random.default_rng(int(params.get("seed", 42)))
        noise_type  = params.get("noise_type", "seamless_perlin")
        octaves     = int(params.get("octaves", 6))
        persistence = float(params.get("persistence", 0.5))
        raw_lac     = float(params.get("lacunarity", 2.0))
        lacunarity  = float(max(0, min(5, round(raw_lac))))
        periods     = int(max(1, round(float(params.get("periods", 3)))))
        turbulence  = float(params.get("turbulence", 0.0))
        color_mode  = params.get("color_mode", "threshold")
        transparent = bool(params.get("transparent_bg", False))

        if noise_type == "seamless_simplex":
            field = self._simplex_field(width, height, periods, octaves,
                                        persistence, lacunarity, turbulence, rng)
        else:
            field = self._perlin_field(width, height, periods, octaves,
                                       persistence, lacunarity, turbulence, rng)

        mn, mx = field.min(), field.max()
        field  = (field - mn) / (mx - mn + 1e-8)

        n = max(1, len(colors))
        bg_idx, exclude = get_bg_params(params, n)
        return _colorise(field, colors, color_mode, transparent, bg_idx, exclude)

    # ── seamless simplex via 4D torus ─────────────────────────────────────────

    def _simplex_field(self, width, height, periods, octaves,
                       persistence, lacunarity, turbulence, rng):
        """
        Map canvas → 4D torus → opensimplex noise4 (scalar, called per pixel).
        Perfectly seamless by construction; no axis-aligned artefacts.
        Falls back to perlin if opensimplex is not installed.
        """
        try:
            import opensimplex as osx
        except ImportError:
            return self._perlin_field(width, height, periods, octaves,
                                      persistence, lacunarity, turbulence, rng)

        # Torus radius: one canvas period = one torus revolution
        R = periods / (2.0 * math.pi)

        # Pre-compute torus coordinates for each row and column
        xs = np.linspace(0.0, 2 * math.pi * periods / periods, width,  endpoint=False)
        ys = np.linspace(0.0, 2 * math.pi * periods / periods, height, endpoint=False)
        # Full 2π range * periods for the correct number of repetitions
        xs = np.linspace(0.0, 2 * math.pi, width,  endpoint=False) * periods
        ys = np.linspace(0.0, 2 * math.pi, height, endpoint=False) * periods

        cx_arr = np.cos(xs) * R;  sx_arr = np.sin(xs) * R   # (W,)
        cy_arr = np.cos(ys) * R;  sy_arr = np.sin(ys) * R   # (H,)

        # Random offsets per octave (4 values each)
        offsets = rng.uniform(0, 100, size=(max(octaves, 3), 4))

        field = np.zeros((height, width), dtype=np.float32)
        amp   = 1.0
        freq  = 1.0
        noise4 = osx.noise4  # local alias for speed

        for k in range(octaves):
            ox, oy, oz, ow = (float(v) for v in offsets[k])
            amp_k = float(amp)
            freq_k = float(freq)
            for j in range(height):
                cz = float(cy_arr[j]) * freq_k + oz
                sw = float(sy_arr[j]) * freq_k + ow
                for i in range(width):
                    field[j, i] += noise4(
                        float(cx_arr[i]) * freq_k + ox,
                        float(sx_arr[i]) * freq_k + oy,
                        cz, sw,
                    ) * amp_k
            amp  *= persistence
            freq *= lacunarity

        if turbulence > 0.0:
            # Warp coordinates with a second simplex field, then re-evaluate
            warp_off = offsets + 500.0
            wx = np.zeros((height, width), dtype=np.float32)
            wy = np.zeros((height, width), dtype=np.float32)
            amp2  = float(turbulence)
            freq2 = 1.0
            for k in range(max(1, octaves // 2)):
                ox2, oy2, oz2, ow2 = (float(v) for v in warp_off[k])
                ox3, oy3, oz3, ow3 = ox2+50., oy2+50., oz2+50., ow2+50.
                amp2k = amp2; freq2k = float(freq2)
                for j in range(height):
                    cz2 = float(cy_arr[j]) * freq2k + oz2
                    sw2 = float(sy_arr[j]) * freq2k + ow2
                    cz3 = float(cy_arr[j]) * freq2k + oz3
                    sw3 = float(sy_arr[j]) * freq2k + ow3
                    for i in range(width):
                        bx = float(cx_arr[i]) * freq2k
                        by = float(sx_arr[i]) * freq2k
                        wx[j, i] += noise4(bx+ox2, by+oy2, cz2, sw2) * amp2k
                        wy[j, i] += noise4(bx+ox3, by+oy3, cz3, sw3) * amp2k
                amp2  *= persistence
                freq2 *= lacunarity
            # Re-evaluate with warped torus angles
            field2 = np.zeros((height, width), dtype=np.float32)
            amp3 = 1.0; freq3 = 1.0
            for k in range(octaves):
                ox4, oy4, oz4, ow4 = (float(v) for v in offsets[k])
                amp3k = float(amp3); freq3k = float(freq3)
                for j in range(height):
                    for i in range(width):
                        # Warp the angle on the torus
                        ax = xs[i] + float(wx[j, i])
                        ay = ys[j] + float(wy[j, i])
                        field2[j, i] += noise4(
                            math.cos(ax) * R * freq3k + ox4,
                            math.sin(ax) * R * freq3k + oy4,
                            math.cos(ay) * R * freq3k + oz4,
                            math.sin(ay) * R * freq3k + ow4,
                        ) * amp3k
                amp3  *= persistence
                freq3 *= lacunarity
            field = field2

        return field

    # ── seamless Perlin ────────────────────────────────────────────────────────

    def _perlin_field(self, width, height, periods, octaves,
                      persistence, lacunarity, turbulence, rng):
        import noise as noise_lib

        base_x = float(rng.integers(0, periods * 100)) / 100.0
        base_y = float(rng.integers(0, periods * 100)) / 100.0

        field = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            ny = (y / height) * periods
            for x in range(width):
                nx = (x / width) * periods
                nx2, ny2 = nx + base_x, ny + base_y

                if turbulence > 0.0:
                    wx = noise_lib.pnoise2(
                        nx2 + 13.1, ny2 + 7.7,
                        octaves=max(1, octaves // 2),
                        persistence=persistence, lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods)
                    wy = noise_lib.pnoise2(
                        nx2 + 3.3, ny2 + 17.2,
                        octaves=max(1, octaves // 2),
                        persistence=persistence, lacunarity=lacunarity,
                        repeatx=periods, repeaty=periods)
                    nx2 += turbulence * wx
                    ny2 += turbulence * wy

                field[y, x] = noise_lib.pnoise2(
                    nx2, ny2,
                    octaves=octaves, persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=periods, repeaty=periods)
        return field
