"""
Collage Generator – stamp PNG/JPG shapes across the canvas.

Shape loading:
  • PNG files are loaded with alpha channel if present.
  • JPG/JPEG files are loaded as BGR and converted to BGRA (alpha = 255 everywhere,
    i.e., fully opaque stamps; use tint_strength to recolour them).
  • Both are tinted toward a random palette colour.

When `transparent_bg` is True the canvas starts fully transparent (BGRA).
Only the shape stamps are drawn opaque, making this suitable as a second layer.
"""
from __future__ import annotations
import math
import os
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}


def _load_shapes(folder: str) -> list[np.ndarray]:
    """Load all PNG/JPG files from folder as BGRA uint8 arrays."""
    shapes = []
    if not folder or not os.path.isdir(folder):
        return shapes
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue
        path = os.path.join(folder, fname)
        if ext == ".png":
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        else:  # jpg / jpeg
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = 255  # fully opaque
        shapes.append(img)
    return shapes


def _tint_bgra(bgra: np.ndarray, target_rgb: tuple, strength: float) -> np.ndarray:
    """
    Tint the colour channels of a float32 BGRA image (values in [0,1])
    toward target_rgb, preserving luminance.
    """
    tr, tg, tb = [c / 255.0 for c in target_rgb]
    b, g, r = bgra[:,:,0], bgra[:,:,1], bgra[:,:,2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    tinted_b = lum * tb
    tinted_g = lum * tg
    tinted_r = lum * tr
    out = bgra.copy()
    out[:,:,0] = b * (1 - strength) + tinted_b * strength
    out[:,:,1] = g * (1 - strength) + tinted_g * strength
    out[:,:,2] = r * (1 - strength) + tinted_r * strength
    return out


def _blend(canvas_f, layer_f, alpha_f, mode):
    """Blend layer onto canvas using given mode. All float32 in [0,1]."""
    if mode == "multiply":
        blended = canvas_f * layer_f
    elif mode == "screen":
        blended = 1.0 - (1.0 - canvas_f) * (1.0 - layer_f)
    elif mode == "overlay":
        blended = np.where(canvas_f < 0.5,
                           2.0 * canvas_f * layer_f,
                           1.0 - 2.0 * (1.0 - canvas_f) * (1.0 - layer_f))
    else:
        blended = layer_f
    return canvas_f * (1.0 - alpha_f) + blended * alpha_f


class CollageGenerator(BaseGenerator):
    name = "Collage"
    description = (
        "Stamp PNG/JPG shapes across the canvas with palette tinting. "
        "Enable 'Transparent background' when using as a second layer."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["collage"]

    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],
        params: dict,
    ) -> np.ndarray:
        shape_folder  = str(params.get("shape_folder", ""))
        count         = int(params.get("count", 40))
        scale_min     = float(params.get("scale_min", 0.05))
        scale_max     = float(params.get("scale_max", 0.25))
        rot_range     = float(params.get("rotation_range", 180))
        blend_mode    = params.get("blend_mode", "normal")
        tint_strength = float(params.get("tint_strength", 0.8))
        transparent   = bool(params.get("transparent_bg", False))
        seed          = int(params.get("seed", 42))

        rng  = np.random.default_rng(seed)
        n    = max(1, len(colors))
        base = min(width, height)

        # ── Canvas ────────────────────────────────────────────────────────────
        if transparent:
            # BGRA, bg fully transparent
            canvas_f = np.zeros((height, width, 4), dtype=np.float32)
        else:
            r0, g0, b0 = colors[0]
            canvas_f = np.full((height, width, 3),
                               [b0/255.0, g0/255.0, r0/255.0], dtype=np.float32)

        # ── Load shapes ───────────────────────────────────────────────────────
        png_shapes = _load_shapes(shape_folder)
        use_png    = len(png_shapes) > 0

        for _ in range(count):
            ci    = int(rng.integers(0, n))
            tint  = colors[ci]
            sz    = int(rng.uniform(scale_min, scale_max) * base)
            sz    = max(8, sz)
            cx    = int(rng.integers(0, width))
            cy    = int(rng.integers(0, height))
            angle = rng.uniform(-rot_range, rot_range)

            raw = (png_shapes[int(rng.integers(0, len(png_shapes)))]
                   if use_png else self._make_procedural(sz, rng))

            self._stamp(canvas_f, raw, cx, cy, sz, angle,
                        tint, tint_strength, blend_mode, width, height, transparent)

        if transparent:
            result = (np.clip(canvas_f, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            result = (np.clip(canvas_f, 0.0, 1.0) * 255).astype(np.uint8)
        return result

    # ── stamping ──────────────────────────────────────────────────────────────

    def _stamp(self, canvas_f, shape_bgra, cx, cy, sz, angle,
               tint, tint_strength, blend_mode, W, H, transparent):
        resized = cv2.resize(shape_bgra, (sz, sz), interpolation=cv2.INTER_AREA)
        M       = cv2.getRotationMatrix2D((sz/2, sz/2), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (sz, sz),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0,0))

        bgr_f  = rotated[:,:,:3].astype(np.float32) / 255.0
        a_f    = rotated[:,:, 3].astype(np.float32) / 255.0

        bgr_f  = _tint_bgra(
            np.dstack([bgr_f, a_f]),
            tint, tint_strength
        )[:,:,:3]

        # Toroidal stamp: iterate over wrapped copies
        for dy in (-H, 0, H):
            for dx in (-W, 0, W):
                x0 = cx - sz//2 + dx
                y0 = cy - sz//2 + dy
                x1, y1 = x0+sz, y0+sz

                ix0, iy0 = max(0, x0), max(0, y0)
                ix1, iy1 = min(W, x1), min(H, y1)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue

                sx0, sy0 = ix0-x0, iy0-y0
                sx1, sy1 = sx0+(ix1-ix0), sy0+(iy1-iy0)

                a_roi   = a_f[sy0:sy1, sx0:sx1, np.newaxis]
                l_roi   = bgr_f[sy0:sy1, sx0:sx1]

                if transparent:
                    # For BGRA canvas: blend colour channels and accumulate alpha
                    c_roi = canvas_f[iy0:iy1, ix0:ix1, :3]
                    canvas_f[iy0:iy1, ix0:ix1, :3] = _blend(c_roi, l_roi, a_roi, blend_mode)
                    # Alpha = max of existing and new stamp alpha
                    existing_a = canvas_f[iy0:iy1, ix0:ix1, 3:4]
                    canvas_f[iy0:iy1, ix0:ix1, 3:4] = np.maximum(existing_a, a_roi)
                else:
                    c_roi = canvas_f[iy0:iy1, ix0:ix1]
                    canvas_f[iy0:iy1, ix0:ix1] = _blend(c_roi, l_roi, a_roi, blend_mode)

    # ── built-in procedural shapes ────────────────────────────────────────────

    def _make_procedural(self, size: int, rng: np.random.Generator) -> np.ndarray:
        sz  = max(8, size)
        img = np.zeros((sz, sz, 4), dtype=np.uint8)
        cx, cy, r = sz//2, sz//2, sz//2 - 2
        shape = int(rng.integers(0, 4))
        white = (255, 255, 255, 255)

        if shape == 0:
            rx = max(2, int(rng.uniform(0.3, 1.0) * r))
            ry = max(2, int(rng.uniform(0.3, 1.0) * r))
            cv2.ellipse(img, (cx,cy), (rx,ry), 0, 0, 360, white, -1)
        elif shape == 1:
            n_pts  = int(rng.integers(5, 9))
            angles = np.sort(rng.uniform(0, 2*math.pi, n_pts))
            radii  = rng.uniform(r*0.4, r*0.95, n_pts)
            pts    = np.array([
                [int(cx+radii[i]*math.cos(angles[i])),
                 int(cy+radii[i]*math.sin(angles[i]))]
                for i in range(n_pts)
            ], np.int32).reshape(-1,1,2)
            cv2.fillPoly(img, [pts], white)
        elif shape == 2:
            angle = rng.uniform(0, 180)
            rect  = ((cx,cy),(int(r*rng.uniform(0.5,1.8)),int(r*rng.uniform(0.3,0.9))),angle)
            box   = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(img, [box], white)
        else:
            cv2.circle(img, (cx,cy), r, white, -1)
        return img
