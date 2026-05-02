"""
African Pattern Generator – perfectly seamless traditional African textile motifs.

Seamless strategy
─────────────────
  1. Find cell dimensions cs_w, cs_h that EXACTLY DIVIDE W and H via
     _nearest_divisor().  This guarantees n_cols * cs_w == W exactly.
  2. Draw on a 3W × 3H canvas with loops covering all 3×n_rows and 3×n_cols cells.
  3. Crop the centre tile [H:2H, W:2W].  Since cell sizes divide W and H exactly,
     every column starts and finishes on a whole cell boundary → perfectly seamless.

Colour seamlessness
───────────────────
Cell colour is _cell_color(row % n_rows, col % n_cols, ...) — purely positional,
periodic with (n_rows, n_cols) → seamless in both x and y.

fill_density controls how many cells receive detailed infill (0=plain, 1=all filled).
Background colour is always excluded from drawn elements.
Seed randomises the foreground colour assignment order.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator, get_bg_params, apply_transparent_bg
from config.defaults import GENERATORS


def _nearest_divisor(container: int, target: int) -> int:
    """Return the divisor of `container` closest to `target` (≥1)."""
    if target >= container:
        return container
    best, best_d = 1, abs(container - target)
    for d in range(1, container + 1):
        if container % d == 0:
            dist = abs(d - target)
            if dist < best_d:
                best, best_d = d, dist
    return best


def _fg_cycle(colors_bgr: list, bg_idx: int, rng) -> list:
    fg = [c for i, c in enumerate(colors_bgr) if i != bg_idx]
    if not fg:
        fg = list(colors_bgr)
    idx = list(range(len(fg)))
    rng.shuffle(idx)
    return [fg[i] for i in idx]


def _cell_color(row, col, nr, nc, fg, offset=0):
    return fg[(offset + row * nc + col) % len(fg)]


# ── kente ──────────────────────────────────────────────────────────────────────

def _draw_kente(canvas, CW, CH, W, H, cs_w, band_h, nr, nc, fg, lw, ci, fill_density, rng):
    nf = len(fg)
    for row in range(3 * nr):
        y  = row * band_h
        bc = _cell_color(row % nr, 0, nr, 1, fg, ci)
        cv2.rectangle(canvas, (0, y), (CW - 1, y + band_h - 1), bc, -1)
        for col in range(3 * nc):
            if rng.random() >= fill_density:
                continue
            x  = col * cs_w
            fc = _cell_color(row % nr, col % nc, nr, nc, fg, ci + 1)
            ac = _cell_color(row % nr, col % nc, nr, nc, fg, ci + 2)
            st = int(rng.integers(0, 4))
            cw, ch = cs_w, band_h
            if st == 0:
                cv2.line(canvas, (x, y), (x + cw, y + ch), ac, lw)
                cv2.line(canvas, (x + cw, y), (x, y + ch), ac, lw)
            elif st == 1:
                step = max(2, cw // 4)
                for i in range(x, x + cw, step):
                    cv2.line(canvas, (i, y), (i, y + ch), fc, max(1, lw - 1))
            elif st == 2:
                step = max(2, ch // 4)
                for j in range(y, y + ch, step):
                    cv2.line(canvas, (x, j), (x + cw, j), fc, max(1, lw - 1))
            else:
                cv2.rectangle(canvas, (x, y), (x + cw - 1, y + ch - 1), fc, lw)


# ── kuba ───────────────────────────────────────────────────────────────────────

def _draw_kuba(canvas, CW, CH, W, H, cs, nr, nc, fg, lw, ci, fill_density, rng):
    for row in range(3 * nr):
        for col in range(3 * nc):
            x  = col * cs; y = row * cs
            c1 = _cell_color(row % nr, col % nc,     nr, nc, fg, ci)
            c2 = _cell_color(row % nr, (col+1) % nc, nr, nc, fg, ci + 1)
            cv2.rectangle(canvas, (x, y), (x + cs - 1, y + cs - 1), c1, -1)
            if rng.random() < fill_density:
                step = max(2, cs // 4)
                for i, s in enumerate(range(0, cs, step)):
                    c = c2 if i % 2 == 0 else c1
                    pts = np.array([
                        [x+s, y], [x+s+step, y], [x+s+step, y+cs//2],
                        [x+cs, y+cs//2], [x+cs, y+cs], [x+s, y+cs],
                    ], dtype=np.int32)
                    cv2.polylines(canvas, [pts], False, c, lw)


# ── adinkra ────────────────────────────────────────────────────────────────────

def _gye_nyame(canvas, cx, cy, r, color, lw):
    cv2.ellipse(canvas, (cx, cy), (r, int(r * 1.3)), 0, 0, 360, color, lw)
    cv2.ellipse(canvas, (cx, cy), (max(2, r*3//5), max(2, r*3//5)), 0, 0, 360, color, lw)
    cv2.line(canvas, (cx - r, cy), (cx + r, cy), color, lw)
    cv2.line(canvas, (cx, cy - int(r*1.3)), (cx, cy + int(r*1.3)), color, lw)
    dr = max(1, r // 5)
    for dx, dy in [(-r//2,-r//2),(r//2,-r//2),(-r//2,r//2),(r//2,r//2)]:
        cv2.circle(canvas, (cx+dx, cy+dy), dr, color, -1)


def _draw_adinkra(canvas, CW, CH, W, H, cs_w, cs_h, nr, nc, fg, lw, ci, fill_density, rng):
    r = max(4, min(cs_w, cs_h) * 2 // 5)
    hx = cs_w // 2; hy = cs_h // 2
    for row in range(3 * nr):
        for col in range(3 * nc):
            if rng.random() >= fill_density:
                continue
            cx    = col * cs_w + hx
            cy    = row * cs_h + hy
            color = _cell_color(row % nr, col % nc, nr, nc, fg, ci + 1)
            _gye_nyame(canvas, cx, cy, r, color, lw)


# ── mudcloth ────────────────────────────────────────────────────────────────────

def _draw_mudcloth(canvas, CW, CH, W, H, cs_w, band_h, nr, nc, fg, lw, ci, fill_density, rng):
    for row in range(3 * nr):
        for col in range(3 * nc):
            x    = col * cs_w; y = row * band_h
            bg_c = _cell_color(row % nr, col % nc,     nr, nc, fg, ci)
            fgc  = _cell_color(row % nr, (col+1) % nc, nr, nc, fg, ci + 1)
            cv2.rectangle(canvas, (x, y), (x + cs_w - 1, y + band_h - 1), bg_c, -1)
            if rng.random() >= fill_density:
                continue
            pat = (row % nr * nc + col % nc) % 4
            if pat == 0:
                step = max(2, (cs_w + band_h) // 8)
                for k in range(-band_h, cs_w + band_h, step):
                    cv2.line(canvas, (x+k, y), (x+k+band_h, y+band_h), fgc, lw)
            elif pat == 1:
                step = max(2, (cs_w + band_h) // 8)
                for k in range(-band_h, cs_w + band_h, step):
                    cv2.line(canvas, (x+k+band_h, y), (x+k, y+band_h), fgc, lw)
            elif pat == 2:
                step = max(2, cs_w // 5)
                for i in range(x, x + cs_w, step):
                    cv2.line(canvas, (i, y), (i, y + band_h), fgc, lw)
                for j in range(y, y + band_h, step):
                    cv2.line(canvas, (x, j), (x + cs_w, j), fgc, lw)
            else:
                mx, my = x + cs_w // 2, y + band_h // 2
                cv2.line(canvas, (x, y),       (mx, y),            fgc, lw)
                cv2.line(canvas, (mx, y),       (mx, my),           fgc, lw)
                cv2.line(canvas, (mx, my),      (x+cs_w, my),       fgc, lw)
                cv2.line(canvas, (x+cs_w, my),  (x+cs_w, y+band_h), fgc, lw)


# ── generator ──────────────────────────────────────────────────────────────────

class AfricanPatternGenerator(BaseGenerator):
    name = "African Pattern"
    description = (
        "African textile patterns: kente, kuba, adinkra, mudcloth. "
        "Perfectly seamless: cell sizes snap to exact divisors of W and H via "
        "_nearest_divisor; drawn on 3×3 canvas and centre-cropped. "
        "Bg colour always excluded. fill_density and seed both work."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["african_pattern"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        motif        = params.get("motif", "kente")
        cell_size    = max(4, int(params.get("cell_size",    48)))
        band_height  = max(4, int(params.get("band_height",  32)))
        fill_density = float(params.get("fill_density",      0.5))
        lw           = max(1, int(params.get("line_width",    2)))
        rotation     = int(params.get("rotation", 0)) % 4
        transparent  = bool(params.get("transparent_bg", False))
        seed         = int(params.get("seed", 42))

        rng    = np.random.default_rng(seed)
        n      = max(1, len(colors))
        bg_idx, _ = get_bg_params(params, n)
        W, H   = width, height
        if rotation % 2 == 1:
            W, H = H, W

        colors_bgr = [(int(b), int(g), int(r)) for r, g, b in colors]
        bg_c       = colors_bgr[bg_idx]
        fg         = _fg_cycle(colors_bgr, bg_idx, rng)
        nf         = len(fg)
        ci         = int(rng.integers(0, nf))

        # Snap sizes to EXACT divisors so n_cols * cs_w == W exactly
        if motif == "kuba":
            # Square cells: need same cs for both W and H
            cs_sq = _nearest_divisor(W, cell_size)
            # Also ensure it divides H; try nearby values
            cs_w = cs_h = cs_sq
            if H % cs_w != 0:
                # Search for cs that divides both W and H
                best_cs = cs_sq; best_d = abs(cs_sq - cell_size)
                for c in range(max(1, cell_size - cell_size//2), cell_size + cell_size//2 + 1):
                    if W % c == 0 and H % c == 0:
                        d = abs(c - cell_size)
                        if d < best_d:
                            best_d = d; best_cs = c
                cs_w = cs_h = best_cs if best_cs > 0 else _nearest_divisor(W, cell_size)
            nc = W // cs_w; nr = H // cs_h
            band_h = cs_h  # for kuba, band_h == cs_h
        elif motif == "adinkra":
            cs_w = _nearest_divisor(W, cell_size)
            cs_h = _nearest_divisor(H, cell_size)
            nc   = W // cs_w; nr = H // cs_h
            band_h = cs_h
        else:
            # kente, mudcloth: separate width and height divisors
            cs_w   = _nearest_divisor(W, cell_size)
            band_h = _nearest_divisor(H, band_height)
            nc     = W // cs_w; nr = H // band_h
            cs_h   = band_h

        CW, CH = 3 * W, 3 * H
        canvas = np.zeros((CH, CW, 3), dtype=np.uint8)
        canvas[:] = bg_c

        if motif == "kente":
            _draw_kente(canvas, CW, CH, W, H, cs_w, band_h, nr, nc,
                        fg, lw, ci, fill_density, rng)
        elif motif == "kuba":
            _draw_kuba(canvas, CW, CH, W, H, cs_w, nr, nc,
                       fg, lw, ci, fill_density, rng)
        elif motif == "adinkra":
            _draw_adinkra(canvas, CW, CH, W, H, cs_w, cs_h, nr, nc,
                          fg, lw, ci, fill_density, rng)
        else:  # mudcloth
            _draw_mudcloth(canvas, CW, CH, W, H, cs_w, band_h, nr, nc,
                           fg, lw, ci, fill_density, rng)

        # Crop centre tile — seamless by exact-divisor construction
        result = canvas[H:2*H, W:2*W].copy()

        if rotation == 1:   result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 2: result = cv2.rotate(result, cv2.ROTATE_180)
        elif rotation == 3: result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if result.shape[0] != height or result.shape[1] != width:
            result = cv2.resize(result, (width, height), interpolation=cv2.INTER_NEAREST)

        if transparent:
            return apply_transparent_bg(result, colors, bg_idx)
        return result
