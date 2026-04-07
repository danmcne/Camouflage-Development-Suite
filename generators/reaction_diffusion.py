"""
Reaction-Diffusion Generator – Gray-Scott two-chemical system.

Simulation runs at a reduced `work_size` (default 200×200) then is upscaled
to the requested output size. This keeps the simulation fast while producing
rich, organic patterns.

Pattern types are controlled primarily by (feed, kill):
  Spots      feed≈0.035, kill≈0.065  – isolated blobs
  Labyrinth  feed≈0.037, kill≈0.060  – connected maze channels
  Stripes    feed≈0.060, kill≈0.062  – elongated stripes
  Coral      feed≈0.062, kill≈0.061  – coral / fingerprint branching
  Mitosis    feed≈0.028, kill≈0.062  – dividing cells / leopard spots

Select a preset from the combo and the feed/kill sliders update automatically.
Choose "Custom" to tune freely.

Anisotropy > 1 stretches patterns horizontally (towards stripes);
Anisotropy < 1 stretches vertically.

Seamless output: the simulation uses toroidal (wrap-around) boundary conditions
so the result tiles perfectly.
"""
from __future__ import annotations
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS, RD_PRESETS


class ReactionDiffusionGenerator(BaseGenerator):
    name = "Reaction-Diffusion"
    description = (
        "Gray-Scott two-chemical reaction-diffusion. "
        "Choose a preset (Spots / Labyrinth / Stripes / Coral / Mitosis) "
        "or fine-tune feed + kill manually. Toroidal (seamless) boundaries."
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
        seed       = int(params.get("seed",           42))

        # Apply preset if not Custom
        if preset != "Custom" and preset in RD_PRESETS:
            feed, kill = RD_PRESETS[preset]
        else:
            feed = float(params.get("feed", 0.037))
            kill = float(params.get("kill", 0.060))

        rng = np.random.default_rng(seed)

        W = work_size
        H = work_size

        # ── Initialise ────────────────────────────────────────────────────────
        U = np.ones((H, W), dtype=np.float32)
        V = np.zeros((H, W), dtype=np.float32)

        # Seed a small noisy blob in the centre
        r = max(4, W // 8)
        cx, cy = W // 2, H // 2
        U[cy - r:cy + r, cx - r:cx + r] = 0.5
        V[cy - r:cy + r, cx - r:cx + r] = 0.25
        # Sprinkle random seeds for richer initial conditions
        n_extra = max(1, W // 20)
        for _ in range(n_extra):
            ex = int(rng.integers(0, W))
            ey = int(rng.integers(0, H))
            er = max(2, W // 16)
            U[max(0,ey-er):ey+er, max(0,ex-er):ex+er] = 0.5
            V[max(0,ey-er):ey+er, max(0,ex-er):ex+er] += rng.uniform(0.2, 0.4)

        V += rng.uniform(0, noise_amp, (H, W)).astype(np.float32)
        np.clip(V, 0, 1, out=V)
        np.clip(U, 0, 1, out=U)

        dt = 1.0

        # ── Simulation (toroidal via np.roll) ─────────────────────────────────
        for step in range(time_steps):
            # Laplacian with toroidal (wrap) boundary
            lap_U = (
                np.roll(U,  1, axis=0) + np.roll(U, -1, axis=0) +
                np.roll(U,  1, axis=1) / anisotropy +
                np.roll(U, -1, axis=1) / anisotropy -
                4.0 * U
            )
            lap_V = (
                np.roll(V,  1, axis=0) + np.roll(V, -1, axis=0) +
                np.roll(V,  1, axis=1) * anisotropy +
                np.roll(V, -1, axis=1) * anisotropy -
                4.0 * V
            )
            uvv = U * V * V
            U += dt * (Du * lap_U - uvv + feed * (1.0 - U))
            V += dt * (Dv * lap_V + uvv - (feed + kill) * V)
            np.clip(U, 0.0, 1.0, out=U)
            np.clip(V, 0.0, 1.0, out=V)

        # ── Quantise V into colour bands ──────────────────────────────────────
        n = max(1, len(colors))
        vmin, vmax = V.min(), V.max()
        if vmax > vmin:
            V_norm = (V - vmin) / (vmax - vmin)
        else:
            V_norm = np.zeros_like(V)

        canvas_small = np.zeros((H, W, 3), dtype=np.uint8)
        thresholds = np.linspace(0.0, 1.0 + 1e-6, n + 1)
        for i, (r_c, g_c, b_c) in enumerate(colors):
            mask = (V_norm >= thresholds[i]) & (V_norm < thresholds[i + 1])
            canvas_small[mask] = (int(b_c), int(g_c), int(r_c))

        # Upscale to requested size
        if (W, H) != (width, height):
            canvas = cv2.resize(canvas_small, (width, height),
                                interpolation=cv2.INTER_NEAREST)
        else:
            canvas = canvas_small

        return canvas
