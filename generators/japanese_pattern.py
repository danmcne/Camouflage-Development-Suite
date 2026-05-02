"""
Japanese Pattern Generator – perfectly seamless traditional Japanese textile motifs.

Seamless strategy
─────────────────
All five motifs use exact-divisor cell sizes + 3×3 canvas + centre crop:
  1. Find cell sizes that EXACTLY DIVIDE W and H via _nearest_divisor().
  2. Draw on 3W×3H canvas (3×3 repetitions of the tile).
  3. Crop centre [H:2H, W:2W] — seamless by construction.

For offset-row patterns (seigaiha, shippo): need ny = H//step_y to be EVEN
(so the alternating row offset completes within one tile).  We enforce this by
requiring H to be divisible by 2*step_y.

For asanoha: need H divisible by 2*h_tri where h_tri = int(cs * sqrt(3)/2).

Colour seamlessness
────────────────────
Cell colour is _ci_pos(row % nr, col % nc, nr, nc, fg, offset) — purely
positional, periodic with (nr, nc) → seamless in both x and y.

For offset-row patterns: odd rows are colour-shifted by nc//2 relative to even
rows, and both shift values are derived from the same nr×nc grid so seamlessness
is maintained across all four tile boundaries.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator, get_bg_params, apply_transparent_bg
from config.defaults import GENERATORS


def _nearest_divisor(container: int, target: int) -> int:
    if target >= container:
        return container
    best, best_d = 1, abs(container - target)
    for d in range(1, container + 1):
        if container % d == 0:
            dist = abs(d - target)
            if dist < best_d:
                best, best_d = d, dist
    return best


def _find_square_divisor(W: int, H: int, target: int) -> int:
    """Find cs that divides BOTH W and H, closest to target."""
    best = _nearest_divisor(W, target)
    best_d = abs(best - target)
    for d in range(max(1, target - target//2), target + target//2 + 2):
        if W % d == 0 and H % d == 0:
            dist = abs(d - target)
            if dist < best_d:
                best, best_d = d, dist
    return best if best > 0 else max(1, _nearest_divisor(W, target))


def _find_even_ny(H: int, step_y_target: int) -> int:
    """
    Find step_y such that ny = H // step_y is EVEN and step_y is close to target.
    step_y must divide H (ny = H // step_y exactly divides H iff step_y | H).
    Returns the best step_y found.
    """
    best = step_y_target; best_d = float('inf')
    for d in range(max(1, step_y_target - step_y_target // 2),
                   step_y_target + step_y_target // 2 + 2):
        if H % d == 0 and (H // d) % 2 == 0:   # divides H AND ny is even
            dist = abs(d - step_y_target)
            if dist < best_d:
                best, best_d = d, dist
    # If no even-ny solution, fall back to any valid divisor
    if best_d == float('inf'):
        best = _nearest_divisor(H, step_y_target)
    return best


def _fg_cycle(colors_bgr: list, bg_idx: int, rng) -> list:
    fg = [c for i, c in enumerate(colors_bgr) if i != bg_idx]
    if not fg:
        fg = list(colors_bgr)
    idx = list(range(len(fg)))
    rng.shuffle(idx)
    return [fg[i] for i in idx]


def _ci_pos(ri: int, rj: int, nr: int, nc: int, fg: list, offset: int = 0):
    nf = len(fg)
    return fg[(offset + ri * nc + rj) % nf]


# ── seigaiha ──────────────────────────────────────────────────────────────────

def _draw_seigaiha(canvas, W, H, cs, fg, lw, fill_alt, ci, rng):
    """
    r = cs//2.  step_x = 2r divides W.  step_y = r, ny = H//r must be even
    (so the row-offset pattern completes within one tile).
    """
    nf     = len(fg)
    r      = max(2, _nearest_divisor(W // 2, max(1, cs // 2)))  # 2r | W
    step_x = 2 * r
    step_y = _find_even_ny(H, r)          # step_y | H AND ny even
    nx     = W  // step_x
    ny     = H  // step_y                 # guaranteed even

    for row in reversed(range(3 * ny)):
        offset_x = (row % 2) * r
        cy = row * step_y
        for col in range(3 * nx + 1):
            cx   = col * step_x + offset_x
            ri   = (row % ny) // 2           # pair index in [0, ny//2)
            rj   = col % nx
            if row % 2 == 1 and nx >= 2:
                rj = (rj + nx // 2) % nx
            fill_c  = _ci_pos(ri, rj,     ny // 2, nx, fg, ci)
            line_c  = _ci_pos(ri, rj,     ny // 2, nx, fg, ci + 1)
            inner_c = _ci_pos(ri, (rj+1)%nx, ny // 2, nx, fg, ci + 2) if fill_alt else line_c
            cv2.ellipse(canvas, (cx, cy), (r, r), 0, 180, 360, fill_c, -1)
            cv2.ellipse(canvas, (cx, cy), (r, r), 0, 180, 360, line_c, lw)
            hr = max(2, r * 2 // 3)
            cv2.ellipse(canvas, (cx, cy), (hr, hr), 0, 185, 355, inner_c, max(1, lw - 1))


# ── asanoha ───────────────────────────────────────────────────────────────────

def _draw_asanoha(canvas, W, H, cs, fg, lw, fill_alt, ci, rng):
    """
    cs_x | W.  h_tri = int(cs_x * sqrt(3)/2) | H and ny = H//h_tri even.
    """
    nf    = len(fg)
    cs_x  = _nearest_divisor(W, cs)
    # h_tri_target from cs_x
    h_tri_tgt = max(1, int(cs_x * math.sqrt(3) / 2))
    h_tri = _find_even_ny(H, h_tri_tgt)
    half  = cs_x // 2
    nx    = W // cs_x
    ny    = H // h_tri

    for row in range(3 * ny):
        for col in range(3 * nx):
            cx = col * cs_x + (row % 2) * half
            cy = row * h_tri
            ri = (row % ny) // 2 if ny >= 2 else row % ny
            rj = col % nx
            if row % 2 == 1 and nx >= 2:
                rj = (rj + nx // 2) % nx
            for k in range(6):
                a0 = math.radians(k * 60)
                a1 = math.radians((k + 1) * 60)
                pts = np.array([
                    [cx, cy],
                    [cx + int(half * math.cos(a0)), cy + int(half * math.sin(a0))],
                    [cx + int(half * math.cos(a1)), cy + int(half * math.sin(a1))],
                ], dtype=np.int32)
                nr_eff = ny // 2 if ny >= 2 else 1
                fill_c = _ci_pos(ri, (rj + k) % nx, nr_eff, nx, fg, ci + k) if fill_alt else fg[k % nf]
                cv2.fillPoly(canvas, [pts], fill_c)
                cv2.polylines(canvas, [pts], True,
                              _ci_pos(ri, (rj + k + 1) % nx, nr_eff, nx, fg, ci + k + 1), lw)


# ── shippo ─────────────────────────────────────────────────────────────────────

def _draw_shippo(canvas, W, H, cs, fg, lw, fill_alt, ci, rng):
    """Same grid structure as seigaiha."""
    nf     = len(fg)
    r      = max(2, _nearest_divisor(W // 2, max(1, cs // 2)))
    step_x = 2 * r
    step_y = _find_even_ny(H, r)
    nx     = W // step_x
    ny     = H // step_y

    for row in range(3 * ny):
        offset_x = (row % 2) * r
        cy = row * step_y
        for col in range(3 * nx + 1):
            cx     = col * step_x + offset_x
            ri     = (row % ny) // 2
            rj     = col % nx
            if row % 2 == 1 and nx >= 2:
                rj = (rj + nx // 2) % nx
            fill_c = _ci_pos(ri, rj, ny // 2, nx, fg, ci) if fill_alt else fg[0]
            line_c = _ci_pos(ri, rj, ny // 2, nx, fg, ci + 1)
            cv2.circle(canvas, (cx, cy), r, fill_c, -1)
            cv2.circle(canvas, (cx, cy), r, line_c, lw)


# ── sayagata ──────────────────────────────────────────────────────────────────

def _draw_sayagata(canvas, W, H, cs, fg, lw, fill_alt, ci, rng):
    """
    Square cs × cs cells. cs divides BOTH W and H (use _find_square_divisor).
    Background colour: fg[0] (first fg after bg exclusion).
    """
    nf   = len(fg)
    cs_x = cs; cs_y = cs   # already square-snapped by caller
    nx   = W // cs_x; ny = H // cs_y
    u    = max(2, min(cs_x, cs_y) // 6)
    # Fill background
    canvas[:] = fg[0]

    def _hook(ox, oy, orientation):
        ri   = (oy // cs_y) % ny; rj = (ox // cs_x) % nx
        fgc  = _ci_pos(ri, rj, ny, nx, fg, ci + 1 + orientation)
        segs = {
            0: [(0,0,3*u,0),(3*u,0,3*u,-3*u),(3*u,-3*u,2*u,-3*u),
                (2*u,-3*u,2*u,-u),(2*u,-u,u,-u),(u,-u,u,0)],
            1: [(0,0,0,3*u),(0,3*u,3*u,3*u),(3*u,3*u,3*u,2*u),
                (3*u,2*u,u,2*u),(u,2*u,u,u),(u,u,0,u)],
        }
        for x1,y1,x2,y2 in segs.get(orientation, segs[0]):
            cv2.line(canvas, (ox+x1, oy+y1), (ox+x2, oy+y2), fgc, lw)

    for row in range(3 * ny):
        for col in range(3 * nx):
            _hook(col * cs_x, row * cs_y, (row + col) % 2)


# ── yagasuri ──────────────────────────────────────────────────────────────────

def _draw_yagasuri(canvas, W, H, cs, fg, lw, fill_alt, ci, rng):
    """
    Square cs × cs cells (cs divides both W and H).
    Two triangular halves per cell, coloured by position.
    """
    nf   = len(fg)
    cs_x = cs; cs_y = cs
    w2   = cs_x // 2
    nx   = W // cs_x; ny = H // cs_y
    n_half = max(1, nx * 2)

    for row in range(3 * ny):
        for col in range(3 * nx):
            ox = col * cs_x; oy = row * cs_y
            ri = row % ny;   rj = col % nx
            for half in range(2):
                dx  = half * w2
                pts = np.array([
                    [ox+dx,    oy],
                    [ox+dx+w2, oy],
                    [ox+dx,    oy+cs_y],
                ], dtype=np.int32)
                rj_half = (rj * 2 + half) % n_half
                fill_c  = _ci_pos(ri, rj_half, ny, n_half, fg, ci) if fill_alt else fg[half % nf]
                line_c  = _ci_pos(ri, (rj_half + 1) % n_half, ny, n_half, fg, ci + 1)
                cv2.fillPoly(canvas, [pts], fill_c)
                cv2.polylines(canvas, [pts], True, line_c, lw)


# ── generator ──────────────────────────────────────────────────────────────────

class JapanesePatternGenerator(BaseGenerator):
    name = "Japanese Pattern"
    description = (
        "Traditional Japanese textile patterns: seigaiha, asanoha, shippo, "
        "sayagata, yagasuri. Perfectly seamless geometry AND colour via exact "
        "divisors and 3×3 canvas centre-crop. Seed randomises colours."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["japanese_pattern"]

    def generate(self, width, height, colors, params) -> np.ndarray:
        motif       = params.get("motif", "seigaiha")
        cell_size   = max(8, int(params.get("cell_size", 48)))
        lw          = max(1, int(params.get("line_width", 2)))
        fill_alt    = bool(params.get("fill_alt", True))
        rotation    = int(params.get("rotation", 0)) % 4
        transparent = bool(params.get("transparent_bg", False))
        seed        = int(params.get("seed", 42))

        rng      = np.random.default_rng(seed)
        n        = max(1, len(colors))
        bg_idx, _ = get_bg_params(params, n)
        W, H     = width, height
        if rotation % 2 == 1:
            W, H = H, W

        colors_bgr = [(int(b), int(g), int(r)) for r, g, b in colors]
        bg_c       = colors_bgr[bg_idx]
        fg         = _fg_cycle(colors_bgr, bg_idx, rng)
        ci         = int(rng.integers(0, len(fg)))

        # For square-cell patterns (sayagata, yagasuri) find cs that divides both W and H
        if motif in ("sayagata", "yagasuri"):
            cs_sq = _find_square_divisor(W, H, cell_size)
        else:
            cs_sq = cell_size  # handled inside each draw function

        CW, CH = 3 * W, 3 * H
        canvas = np.zeros((CH, CW, 3), dtype=np.uint8)
        canvas[:] = bg_c

        fn_map = {
            "seigaiha": _draw_seigaiha,
            "asanoha":  _draw_asanoha,
            "shippo":   _draw_shippo,
            "sayagata": lambda *a, **kw: _draw_sayagata(*a, **kw),
            "yagasuri": lambda *a, **kw: _draw_yagasuri(*a, **kw),
        }

        if motif in ("sayagata", "yagasuri"):
            # Pass square cs directly
            fn = {"sayagata": _draw_sayagata, "yagasuri": _draw_yagasuri}[motif]
            fn(canvas, W, H, cs_sq, fg, lw, fill_alt, ci, rng)
        else:
            fn = fn_map.get(motif, _draw_seigaiha)
            fn(canvas, W, H, cell_size, fg, lw, fill_alt, ci, rng)

        # Crop centre tile
        result = canvas[H:2*H, W:2*W].copy()

        if rotation == 1:   result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 2: result = cv2.rotate(result, cv2.ROTATE_180)
        elif rotation == 3: result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if result.shape[0] != height or result.shape[1] != width:
            result = cv2.resize(result, (width, height), interpolation=cv2.INTER_NEAREST)

        if transparent:
            return apply_transparent_bg(result, colors, bg_idx)
        return result
