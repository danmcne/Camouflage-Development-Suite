"""
Reaction-Diffusion Generator – Gray-Scott, toroidal, with post-upscale blur
and optional transparent background.
"""
from __future__ import annotations
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS, RD_PRESETS
from generators.blur_sharp import _toroidal_blur_2d


class ReactionDiffusionGenerator(BaseGenerator):
    name = "Reaction-Diffusion"
    description = (
        "Gray-Scott two-chemical system. Choose a preset or set feed+kill manually. "
        "Toroidal (seamless). Post-upscale blur reduces pixelation."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["reaction_diffusion"]

    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],
        params: dict,
    ) -> np.ndarray:
        preset     = params.get("preset", "Labyrinth")
        Du         = float(params.get("Du",          0.16))
        Dv         = float(params.get("Dv",          0.08))
        time_steps = int(params.get("time_steps",   3000))
        work_size  = int(params.get("work_size",     200))
        noise_amp  = float(params.get("noise_scale", 0.05))
        anisotropy = float(params.get("anisotropy",  1.0))
        post_blur  = float(params.get("post_blur",   1.0))
        transparent= bool(params.get("transparent_bg", False))
        seed       = int(params.get("seed",           42))

        if preset != "Custom" and preset in RD_PRESETS:
            feed, kill = RD_PRESETS[preset]
        else:
            feed = float(params.get("feed", 0.037))
            kill = float(params.get("kill", 0.060))

        rng = np.random.default_rng(seed)
        W = H = work_size

        U = np.ones((H, W), dtype=np.float32)
        V = np.zeros((H, W), dtype=np.float32)

        r = max(4, W // 8)
        cx, cy = W // 2, H // 2
        U[cy-r:cy+r, cx-r:cx+r] = 0.5
        V[cy-r:cy+r, cx-r:cx+r] = 0.25
        for _ in range(max(1, W // 20)):
            ex = int(rng.integers(0, W))
            ey = int(rng.integers(0, H))
            er = max(2, W // 16)
            U[max(0,ey-er):ey+er, max(0,ex-er):ex+er]  = 0.5
            V[max(0,ey-er):ey+er, max(0,ex-er):ex+er] += rng.uniform(0.2, 0.4)
        V += rng.uniform(0, noise_amp, (H, W)).astype(np.float32)
        np.clip(V, 0, 1, out=V)
        np.clip(U, 0, 1, out=U)

        dt = 1.0
        for _ in range(time_steps):
            lap_U = (np.roll(U,  1, 0) + np.roll(U, -1, 0) +
                     np.roll(U,  1, 1) / anisotropy +
                     np.roll(U, -1, 1) / anisotropy - 4.0 * U)
            lap_V = (np.roll(V,  1, 0) + np.roll(V, -1, 0) +
                     np.roll(V,  1, 1) * anisotropy +
                     np.roll(V, -1, 1) * anisotropy - 4.0 * V)
            uvv    = U * V * V
            U     += dt * (Du * lap_U - uvv + feed * (1.0 - U))
            V     += dt * (Dv * lap_V + uvv - (feed + kill) * V)
            np.clip(U, 0, 1, out=U)
            np.clip(V, 0, 1, out=V)

        mn, mx = V.min(), V.max()
        field  = (V - mn) / (mx - mn) if mx > mn else np.full_like(V, 0.5)

        # Upscale
        if W != width or H != height:
            u8    = (field * 255).clip(0, 255).astype(np.uint8)
            big   = cv2.resize(u8, (width, height), interpolation=cv2.INTER_NEAREST)
            field = big.astype(np.float32) / 255.0

        if post_blur > 0.1:
            field = _toroidal_blur_2d(field, post_blur, post_blur)
            mn, mx = field.min(), field.max()
            if mx > mn:
                field = (field - mn) / (mx - mn)

        # Colourise
        from generators.blur_sharp import _colorise
        return _colorise(field, colors, "threshold", transparent)
