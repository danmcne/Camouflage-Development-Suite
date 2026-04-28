"""
Collage Generator – stamp PNG/JPG shapes or procedural shapes with palette tinting.

Placement modes
───────────────
random (default): shapes placed at independent random positions.
field_driven:     density/orientation/scale Perlin fields drive placement.
grid:             regular square grid, each cell stamped once (toroidal seam).
triangle_grid:    triangular tiling — alternating up/down triangles covering canvas.

Grid modes are fully toroidal: shapes that extend past one edge appear at the
opposite edge via 9-copy toroidal stamping (inherited from _stamp).

Background colour
─────────────────
bg_color_idx: canvas fill colour; shapes are never tinted to that colour.
transparent_bg: makes the canvas transparent where no shapes are stamped.
"""
from __future__ import annotations
import math
import os
import numpy as np
import cv2
from generators.base import BaseGenerator, get_bg_params, apply_transparent_bg, make_fg_colors
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
            if img.ndim == 2:          img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA); img[:,:,3] = 255
        shapes.append(img)
    return shapes


def _tint(bgra_f, target_rgb, strength):
    b,g,r = bgra_f[:,:,0], bgra_f[:,:,1], bgra_f[:,:,2]
    lum   = 0.2126*r + 0.7152*g + 0.0722*b
    tr,tg,tb = [c/255.0 for c in target_rgb]
    out = bgra_f.copy()
    out[:,:,0] = b*(1-strength) + lum*tb*strength
    out[:,:,1] = g*(1-strength) + lum*tg*strength
    out[:,:,2] = r*(1-strength) + lum*tr*strength
    return out


def _blend(base_f, layer_f, alpha_f, mode):
    if   mode == "multiply": blended = base_f * layer_f
    elif mode == "screen":   blended = 1.0-(1.0-base_f)*(1.0-layer_f)
    elif mode == "overlay":  blended = np.where(base_f<0.5, 2*base_f*layer_f,
                                                1-2*(1-base_f)*(1-layer_f))
    else:                    blended = layer_f
    return base_f*(1-alpha_f) + blended*alpha_f


def _build_fields(width, height, field_scale, seed, rng):
    try:
        import noise as noise_lib
    except ImportError:
        noise_lib = None
    fw = max(16, int(width  * field_scale))
    fh = max(16, int(height * field_scale))
    if noise_lib is not None:
        bx = rng.uniform(0,100); by = rng.uniform(0,100); periods=3
        den = np.zeros((fh,fw),dtype=np.float32)
        ori = np.zeros((fh,fw),dtype=np.float32)
        sca = np.zeros((fh,fw),dtype=np.float32)
        for fy in range(fh):
            ny=(fy/fh)*periods
            for fx in range(fw):
                nx=(fx/fw)*periods
                den[fy,fx]=noise_lib.pnoise2(nx+bx,ny+by,octaves=4,persistence=0.5,lacunarity=2,repeatx=periods,repeaty=periods)
                ori[fy,fx]=noise_lib.pnoise2(nx+bx+50,ny+by+37,octaves=3,persistence=0.5,lacunarity=2,repeatx=periods,repeaty=periods)
                sca[fy,fx]=noise_lib.pnoise2(nx+bx+73,ny+by+19,octaves=3,persistence=0.5,lacunarity=2,repeatx=periods,repeaty=periods)
    else:
        k=max(3,(min(fh,fw)//3)|1)
        den=cv2.GaussianBlur(rng.random((fh,fw)).astype(np.float32),(k,k),k/6)
        ori=cv2.GaussianBlur(rng.random((fh,fw)).astype(np.float32),(k,k),k/6)
        sca=cv2.GaussianBlur(rng.random((fh,fw)).astype(np.float32),(k,k),k/6)
    def _n(a):
        lo,hi=a.min(),a.max()
        return (a-lo)/(hi-lo+1e-8)
    return (cv2.resize(_n(den),(width,height),interpolation=cv2.INTER_LINEAR),
            cv2.resize(_n(ori),(width,height),interpolation=cv2.INTER_LINEAR),
            cv2.resize(_n(sca),(width,height),interpolation=cv2.INTER_LINEAR))


def _importance_sample(density, density_contrast, count, rng):
    h,w=density.shape
    weights=(density.ravel()**max(1.0,1+density_contrast*8))
    total=weights.sum()
    if total<=0: weights=np.ones_like(weights); total=weights.sum()
    probs=weights/total
    flat=rng.choice(len(probs),size=count,replace=True,p=probs)
    ys,xs=np.unravel_index(flat,(h,w))
    return list(zip(xs.tolist(),ys.tolist()))


class CollageGenerator(BaseGenerator):
    name = "Collage"
    description = (
        "Stamp shapes with palette tinting. Placement: random, field-driven, "
        "regular grid, or triangular grid. All modes are toroidal."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["collage"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        shape_folder   = str(params.get("shape_folder",    ""))
        count          = int(params.get("count",           40))
        scale_min      = float(params.get("scale_min",     0.05))
        scale_max      = float(params.get("scale_max",     0.25))
        rot_range      = float(params.get("rotation_range",180))
        blend_mode     = params.get("blend_mode",          "normal")
        tint_strength  = float(params.get("tint_strength", 0.8))
        placement      = params.get("placement_mode",      "random")
        grid_cell      = int(params.get("grid_cell_size",  60))
        use_fields     = bool(params.get("use_fields",     False)) and placement=="random"
        field_scale    = float(params.get("field_scale",   0.25))
        density_contrast=float(params.get("density_contrast",0.7))
        orient_coh     = float(params.get("orient_coherence",0.6))
        scale_var      = float(params.get("scale_variation",0.5))
        transparent    = bool(params.get("transparent_bg", False))
        seed           = int(params.get("seed",            42))

        rng  = np.random.default_rng(seed)
        n    = max(1, len(colors))
        base = min(width, height)
        bg_idx, exclude = get_bg_params(params, n)
        scale_min, scale_max = min(scale_min,scale_max), max(scale_min,scale_max)

        fg_colors = make_fg_colors(colors, bg_idx, True)
        if not fg_colors: fg_colors = colors

        if transparent:
            canvas_f = np.zeros((height, width, 4), dtype=np.float32)
        else:
            rb,gb,bb = colors[bg_idx]
            canvas_f = np.full((height, width, 3),
                               [bb/255.0,gb/255.0,rb/255.0], dtype=np.float32)

        png_shapes = _load_shapes(shape_folder)
        use_png    = len(png_shapes) > 0

        def _pick_shape(sz):
            if use_png:
                return png_shapes[int(rng.integers(0,len(png_shapes)))]
            return self._make_procedural(sz, rng)

        def _stamp_one(cx, cy, sz, angle):
            ci   = int(rng.integers(0, len(fg_colors)))
            raw  = _pick_shape(sz)
            self._stamp(canvas_f, raw, cx, cy, sz, angle,
                        fg_colors[ci], tint_strength, blend_mode, width, height, transparent)

        if placement == "grid":
            cell = max(8, grid_cell)
            cols = math.ceil(width  / cell) + 2
            rows = math.ceil(height / cell) + 2
            for row in range(-1, rows):
                for col in range(-1, cols):
                    cx = int((col + 0.5) * cell)
                    cy = int((row + 0.5) * cell)
                    sz    = max(8, int(rng.uniform(scale_min, scale_max) * base))
                    #sz = max(8, int(rng.uniform(scale_min, scale_max) * cell))
                    _stamp_one(cx % width, cy % height, sz,
                               rng.uniform(-rot_range, rot_range))

        elif placement == "triangle_grid":
            cell = max(8, grid_cell)
            cols = math.ceil(width  / cell) + 2
            rows = math.ceil(height / cell) + 2
            for row in range(-1, rows):
                for col in range(-1, cols):
                    # Two triangle centres per cell
                    x0, y0 = col * cell, row * cell
                    # Upper-left triangle centre
                    cx1 = int(x0 + cell // 3)
                    cy1 = int(y0 + cell // 3)
                    # Lower-right triangle centre
                    cx2 = int(x0 + 2 * cell // 3)
                    cy2 = int(y0 + 2 * cell // 3)
                    sz    = max(8, int(rng.uniform(scale_min, scale_max) * base))
                    #sz  = max(8, int(rng.uniform(scale_min, scale_max) * cell))
                    for cx, cy in [(cx1, cy1), (cx2, cy2)]:
                        _stamp_one(cx % width, cy % height, sz,
                                   rng.uniform(-rot_range, rot_range))

        else:  # random or field_driven
            if use_fields:
                density_f, orient_f, scale_f = _build_fields(
                    width, height, field_scale, seed, rng)
                positions = _importance_sample(density_f, density_contrast, count, rng)
            else:
                positions = None

            for i in range(count):
                if use_fields and positions is not None:
                    cx, cy = positions[i]
                    ls = scale_f[cy, cx]
                    sz = max(8, int((scale_min + (scale_max-scale_min)*
                                    ((1-scale_var)*0.5+scale_var*ls)) * base))
                    fa = orient_f[cy, cx] * 360.0 - 180.0
                    ra = rng.uniform(-rot_range, rot_range)
                    angle = fa*orient_coh + ra*(1-orient_coh)
                else:
                    sz    = max(8, int(rng.uniform(scale_min, scale_max) * base))
                    cx    = int(rng.integers(0, width))
                    cy    = int(rng.integers(0, height))
                    angle = rng.uniform(-rot_range, rot_range)
                _stamp_one(cx, cy, sz, angle)

        return (np.clip(canvas_f, 0.0, 1.0) * 255).astype(np.uint8)

    def _stamp(self, canvas_f, shape_bgra, cx, cy, sz, angle,
               tint, tint_strength, blend_mode, W, H, transparent):
        resized = cv2.resize(shape_bgra, (sz, sz), interpolation=cv2.INTER_AREA)
        M       = cv2.getRotationMatrix2D((sz/2,sz/2), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (sz,sz), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        bgr_f   = rotated[:,:,:3].astype(np.float32)/255.0
        a_f     = rotated[:,:,3].astype(np.float32)/255.0
        tinted  = _tint(np.dstack([bgr_f, a_f]), tint, tint_strength)
        bgr_f   = tinted[:,:,:3]

        for dy in (-H,0,H):
            for dx in (-W,0,W):
                x0,y0 = cx-sz//2+dx, cy-sz//2+dy
                x1,y1 = x0+sz, y0+sz
                ix0,iy0 = max(0,x0), max(0,y0)
                ix1,iy1 = min(W,x1), min(H,y1)
                if ix0>=ix1 or iy0>=iy1: continue
                sx0,sy0 = ix0-x0, iy0-y0
                sx1,sy1 = sx0+(ix1-ix0), sy0+(iy1-iy0)
                a_roi = a_f[sy0:sy1,sx0:sx1,np.newaxis]
                l_roi = bgr_f[sy0:sy1,sx0:sx1]
                if transparent:
                    c_roi = canvas_f[iy0:iy1,ix0:ix1,:3]
                    canvas_f[iy0:iy1,ix0:ix1,:3] = _blend(c_roi,l_roi,a_roi,blend_mode)
                    ea = canvas_f[iy0:iy1,ix0:ix1,3:4]
                    canvas_f[iy0:iy1,ix0:ix1,3:4] = np.maximum(ea, a_roi)
                else:
                    c_roi = canvas_f[iy0:iy1,ix0:ix1]
                    canvas_f[iy0:iy1,ix0:ix1] = _blend(c_roi,l_roi,a_roi,blend_mode)

    def _make_procedural(self, size, rng):
        sz  = max(8, size)
        img = np.zeros((sz,sz,4), dtype=np.uint8)
        cx,cy,r = sz//2, sz//2, sz//2-2
        s   = int(rng.integers(0,4))
        W   = (255,255,255,255)
        if s==0:
            rx=max(2,int(rng.uniform(0.3,1.0)*r))
            ry=max(2,int(rng.uniform(0.3,1.0)*r))
            cv2.ellipse(img,(cx,cy),(rx,ry),0,0,360,W,-1)
        elif s==1:
            angs=np.sort(rng.uniform(0,2*math.pi,3))
            radii=rng.uniform(r*0.4,r*0.95,3)
            pts=np.array([[int(cx+radii[i]*math.cos(angs[i])),
                           int(cy+radii[i]*math.sin(angs[i]))]
                          for i in range(3)],np.int32).reshape(-1,1,2)
            cv2.fillPoly(img,[pts],W)
        elif s==2:
            ang=rng.uniform(0,180)
            rect=((cx,cy),(int(r*rng.uniform(0.5,1.8)),int(r*rng.uniform(0.3,0.9))),ang)
            box=cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(img,[box],W)
        else:
            cv2.circle(img,(cx,cy),r,W,-1)
        return img
