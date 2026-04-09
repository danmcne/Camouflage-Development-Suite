"""
Collage Generator – stamp PNG/JPG shapes with palette tinting, toroidal wrapping.

bg_color_idx: the palette colour at this index fills the canvas background.
Shapes are never tinted to that colour, preventing the same-colour outline
composite artefact caused by alpha edges adjacent to identical fills.
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
            if img is None: continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = 255
        shapes.append(img)
    return shapes


def _tint(bgra_f: np.ndarray, target_rgb: tuple, strength: float) -> np.ndarray:
    b, g, r = bgra_f[:,:,0], bgra_f[:,:,1], bgra_f[:,:,2]
    lum     = 0.2126*r + 0.7152*g + 0.0722*b
    tr, tg, tb = [c/255.0 for c in target_rgb]
    out = bgra_f.copy()
    out[:,:,0] = b*(1-strength) + lum*tb*strength
    out[:,:,1] = g*(1-strength) + lum*tg*strength
    out[:,:,2] = r*(1-strength) + lum*tr*strength
    return out


def _blend(base_f, layer_f, alpha_f, mode):
    if   mode == "multiply": blended = base_f * layer_f
    elif mode == "screen":   blended = 1.0 - (1.0-base_f)*(1.0-layer_f)
    elif mode == "overlay":  blended = np.where(base_f<0.5, 2*base_f*layer_f,
                                                1-2*(1-base_f)*(1-layer_f))
    else:                    blended = layer_f
    return base_f*(1-alpha_f) + blended*alpha_f


class CollageGenerator(BaseGenerator):
    name = "Collage"
    description = (
        "Stamp PNG/JPG shapes with palette tinting and toroidal wrapping. "
        "Set 'Background colour index' to avoid same-colour outline artefacts."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["collage"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        shape_folder  = str(params.get("shape_folder",   ""))
        count         = int(params.get("count",          40))
        scale_min     = float(params.get("scale_min",    0.05))
        scale_max     = float(params.get("scale_max",    0.25))
        rot_range     = float(params.get("rotation_range", 180))
        blend_mode    = params.get("blend_mode",         "normal")
        tint_strength = float(params.get("tint_strength",0.8))
        bg_idx        = int(params.get("bg_color_idx",   0))
        transparent   = bool(params.get("transparent_bg",False))
        seed          = int(params.get("seed",           42))

        rng  = np.random.default_rng(seed)
        n    = max(1, len(colors))
        base = min(width, height)
        bg_idx = max(0, min(bg_idx, n - 1))

        # Build foreground colour list (exclude bg colour)
        fg_colors = [c for i, c in enumerate(colors) if i != bg_idx]
        if not fg_colors:
            fg_colors = colors   # fallback if only one colour

        # Canvas
        if transparent:
            canvas_f = np.zeros((height, width, 4), dtype=np.float32)
        else:
            rb, gb, bb = colors[bg_idx]
            canvas_f = np.full((height, width, 3),
                               [bb/255.0, gb/255.0, rb/255.0], dtype=np.float32)

        png_shapes = _load_shapes(shape_folder)
        use_png    = len(png_shapes) > 0

        for _ in range(count):
            ci   = int(rng.integers(0, len(fg_colors)))
            tint = fg_colors[ci]
            sz   = max(8, int(rng.uniform(scale_min, scale_max) * base))
            cx   = int(rng.integers(0, width))
            cy   = int(rng.integers(0, height))
            angle= rng.uniform(-rot_range, rot_range)

            raw  = (png_shapes[int(rng.integers(0, len(png_shapes)))]
                    if use_png else self._make_procedural(sz, rng))

            self._stamp(canvas_f, raw, cx, cy, sz, angle,
                        tint, tint_strength, blend_mode, width, height, transparent)

        return (np.clip(canvas_f, 0.0, 1.0) * 255).astype(np.uint8)

    def _stamp(self, canvas_f, shape_bgra, cx, cy, sz, angle,
               tint, tint_strength, blend_mode, W, H, transparent):
        resized = cv2.resize(shape_bgra, (sz, sz), interpolation=cv2.INTER_AREA)
        M       = cv2.getRotationMatrix2D((sz/2, sz/2), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (sz, sz),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0,0))

        bgr_f = rotated[:,:,:3].astype(np.float32)/255.0
        a_f   = rotated[:,:, 3].astype(np.float32)/255.0

        tinted = _tint(np.dstack([bgr_f, a_f]), tint, tint_strength)
        bgr_f  = tinted[:,:,:3]

        # Toroidal stamp at 9 positions
        for dy in (-H, 0, H):
            for dx in (-W, 0, W):
                x0, y0 = cx-sz//2+dx, cy-sz//2+dy
                x1, y1 = x0+sz,       y0+sz
                ix0, iy0 = max(0,x0), max(0,y0)
                ix1, iy1 = min(W,x1), min(H,y1)
                if ix0>=ix1 or iy0>=iy1: continue
                sx0, sy0 = ix0-x0, iy0-y0
                sx1, sy1 = sx0+(ix1-ix0), sy0+(iy1-iy0)

                a_roi = a_f[sy0:sy1, sx0:sx1, np.newaxis]
                l_roi = bgr_f[sy0:sy1, sx0:sx1]

                if transparent:
                    c_roi = canvas_f[iy0:iy1, ix0:ix1, :3]
                    canvas_f[iy0:iy1, ix0:ix1, :3] = _blend(c_roi, l_roi, a_roi, blend_mode)
                    ea = canvas_f[iy0:iy1, ix0:ix1, 3:4]
                    canvas_f[iy0:iy1, ix0:ix1, 3:4] = np.maximum(ea, a_roi)
                else:
                    c_roi = canvas_f[iy0:iy1, ix0:ix1]
                    canvas_f[iy0:iy1, ix0:ix1] = _blend(c_roi, l_roi, a_roi, blend_mode)

    def _make_procedural(self, size: int, rng) -> np.ndarray:
        sz  = max(8, size)
        img = np.zeros((sz, sz, 4), dtype=np.uint8)
        cx, cy, r = sz//2, sz//2, sz//2-2
        s   = int(rng.integers(0, 4))
        W   = (255,255,255,255)
        if   s == 0:
            rx = max(2,int(rng.uniform(0.3,1.0)*r))
            ry = max(2,int(rng.uniform(0.3,1.0)*r))
            cv2.ellipse(img,(cx,cy),(rx,ry),0,0,360,W,-1)
        elif s == 1:
            n_p   = int(rng.integers(5,9))
            angs  = np.sort(rng.uniform(0,2*math.pi,n_p))
            radii = rng.uniform(r*0.4,r*0.95,n_p)
            pts   = np.array([[int(cx+radii[i]*math.cos(angs[i])),
                               int(cy+radii[i]*math.sin(angs[i]))]
                              for i in range(n_p)],np.int32).reshape(-1,1,2)
            cv2.fillPoly(img,[pts],W)
        elif s == 2:
            ang  = rng.uniform(0,180)
            rect = ((cx,cy),(int(r*rng.uniform(0.5,1.8)),int(r*rng.uniform(0.3,0.9))),ang)
            box  = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(img,[box],W)
        else:
            cv2.circle(img,(cx,cy),r,W,-1)
        return img
