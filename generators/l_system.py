"""
L-System Generator – seamless multi-tree with 9-copy toroidal rendering.

Seam fix: collect ALL segments for each tree, draw at all 9 toroidal
positions. OpenCV clips automatically → perfectly seamless at any canvas size.

Background colour handling: bg_color_idx fills canvas; if exclude_bg_from_elements
is set, tree segments will never use that colour. transparent_bg makes bg-coloured
pixels alpha=0 in the output.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator, get_bg_params, apply_transparent_bg, make_fg_colors
from config.defaults import GENERATORS


def _expand(axiom: str, rules: dict, iterations: int) -> str:
    s = axiom
    for _ in range(iterations):
        s = "".join(rules.get(c, c) for c in s)
    return s


def _parse_rules(rules_str: str) -> dict:
    rules = {}
    for rule in rules_str.split(";"):
        rule = rule.strip()
        if "->" in rule:
            lhs, rhs = rule.split("->", 1)
            rules[lhs.strip()] = rhs.strip()
    return rules


def _collect_segments(sentence, angle_rad, step, start_x, start_y,
                      start_dir, colors, n, color_idx, lw):
    x, y      = start_x, start_y
    direction = start_dir
    ci        = color_idx
    stack     = []
    segs      = []
    for ch in sentence:
        if ch in ("F", "G"):
            nx = x + step * math.cos(direction)
            ny = y + step * math.sin(direction)
            r, g, b = colors[ci % n]
            segs.append((x, y, nx, ny, (int(b), int(g), int(r)), lw))
            x, y = nx, ny
        elif ch == "+": direction += angle_rad
        elif ch == "-": direction -= angle_rad
        elif ch == "[": stack.append((x, y, direction, ci))
        elif ch == "]":
            if stack: x, y, direction, ci = stack.pop()
        elif ch == "f":
            x += step * math.cos(direction)
            y += step * math.sin(direction)
        elif ch == "|": direction += math.pi
    return segs


class LSystemGenerator(BaseGenerator):
    name = "L-System"
    description = (
        "Multiple turtle-based L-System trees with per-tree variation. "
        "Segments drawn at 9 toroidal offsets → perfectly seamless at any size."
    )

    def get_param_schema(self) -> dict:
        return GENERATORS["l_system"]

    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],
        params: dict,
    ) -> np.ndarray:
        axiom       = str(params.get("axiom",    "F"))
        rules_str   = str(params.get("rules",    "F->FF+[+F-F-F]-[-F+F+F]"))
        base_angle  = float(params.get("angle",  25.0))
        angle_var   = float(params.get("angle_var", 8.0))
        iterations  = int(params.get("iterations", 4))
        num_trees   = int(params.get("num_trees",  9))
        step_min    = float(params.get("step_min", 3))
        step_max    = float(params.get("step_max", 8))
        width_min   = int(params.get("width_min", 1))
        width_max   = int(params.get("width_max", 2))
        color_per   = bool(params.get("color_per_tree", True))
        transparent = bool(params.get("transparent_bg",  False))
        seed        = int(params.get("seed", 42))

        rng  = np.random.default_rng(seed)
        n    = max(1, len(colors))
        bg_idx, exclude = get_bg_params(params, n)

        # Foreground colours for tree segments
        fg_colors = make_fg_colors(colors, bg_idx, exclude)
        nf = max(1, len(fg_colors))

        # Canvas — fill with bg colour
        bg_r, bg_g, bg_b = colors[bg_idx]
        if transparent:
            canvas = np.zeros((height, width, 4), dtype=np.uint8)
            # transparent canvas; bg = alpha 0 (already zeros)
        else:
            canvas = np.full((height, width, 3),
                             [int(bg_b), int(bg_g), int(bg_r)], dtype=np.uint8)

        sentence = _expand(axiom, _parse_rules(rules_str), iterations)

        for tree_idx in range(num_trees):
            angle_rad = math.radians(base_angle + rng.uniform(-angle_var, angle_var))
            step = rng.uniform(step_min, step_max)
            lw   = int(rng.integers(width_min, width_max + 1))
            ci   = int(rng.integers(0, nf)) if color_per else tree_idx % nf

            segs = _collect_segments(
                sentence, angle_rad, step,
                rng.uniform(0, width), rng.uniform(0, height),
                rng.uniform(0, 2 * math.pi),
                fg_colors, nf, ci, lw,
            )

            for dy in (-height, 0, height):
                for dx in (-width, 0, width):
                    for x0, y0, x1, y1, color_bgr, line_w in segs:
                        px0, py0 = int(x0+dx), int(y0+dy)
                        px1, py1 = int(x1+dx), int(y1+dy)
                        if transparent:
                            cv2.line(canvas, (px0,py0),(px1,py1), color_bgr+(255,), line_w)
                        else:
                            cv2.line(canvas, (px0,py0),(px1,py1), color_bgr, line_w)

        if transparent:
            return canvas   # alpha=0 where no segments were drawn (the bg)

        return canvas
