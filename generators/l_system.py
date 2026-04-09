"""
L-System Generator – seamless multi-tree with 9-copy toroidal rendering.

Seam fix: instead of detecting and skipping boundary-crossing line segments,
we collect ALL segments for each tree and then draw them at all 9 toroidal
positions (dx, dy) ∈ {-W, 0, W} × {-H, 0, H}.  OpenCV clips lines that fall
outside the canvas automatically, so this is both simple and correct.
The result is perfectly seamless at any canvas size.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator
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
    """
    Walk the L-system sentence and return a list of
    (x0, y0, x1, y1, bgr_tuple, linewidth).
    Coordinates are NOT wrapped — they can be negative or > canvas size.
    """
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
        elif ch == "+":
            direction += angle_rad
        elif ch == "-":
            direction -= angle_rad
        elif ch == "[":
            stack.append((x, y, direction, ci))
        elif ch == "]":
            if stack:
                x, y, direction, ci = stack.pop()
        elif ch == "f":
            x += step * math.cos(direction)
            y += step * math.sin(direction)
        elif ch == "|":
            direction += math.pi

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
        axiom       = str(params.get("axiom", "F"))
        rules_str   = str(params.get("rules", "F->FF+[+F-F-F]-[-F+F+F]"))
        base_angle  = float(params.get("angle", 25.0))
        angle_var   = float(params.get("angle_var", 8.0))
        iterations  = int(params.get("iterations", 4))
        num_trees   = int(params.get("num_trees", 9))
        step_min    = float(params.get("step_min", 3))
        step_max    = float(params.get("step_max", 8))
        width_min   = int(params.get("width_min", 1))
        width_max   = int(params.get("width_max", 2))
        color_per   = bool(params.get("color_per_tree", True))
        transparent = bool(params.get("transparent_bg", False))
        seed        = int(params.get("seed", 42))

        rng    = np.random.default_rng(seed)
        rules  = _parse_rules(rules_str)
        n      = max(1, len(colors))

        # Expand once (shared)
        sentence = _expand(axiom, rules, iterations)

        # Canvas
        channels = 4 if transparent else 3
        canvas   = np.zeros((height, width, channels), dtype=np.uint8)
        if not transparent and colors:
            r0, g0, b0 = colors[0]
            canvas[:] = (int(b0), int(g0), int(r0))

        # Collect and draw all trees
        for tree_idx in range(num_trees):
            angle_rad = math.radians(
                base_angle + rng.uniform(-angle_var, angle_var)
            )
            step = rng.uniform(step_min, step_max)
            lw   = int(rng.integers(width_min, width_max + 1))
            ci   = int(rng.integers(0, n)) if color_per else tree_idx % n

            start_x   = rng.uniform(0, width)
            start_y   = rng.uniform(0, height)
            start_dir = rng.uniform(0, 2 * math.pi)

            segs = _collect_segments(
                sentence, angle_rad, step,
                start_x, start_y, start_dir,
                colors, n, ci, lw,
            )

            # Draw at all 9 toroidal offsets
            for dy in (-height, 0, height):
                for dx in (-width, 0, width):
                    for x0, y0, x1, y1, color_bgr, line_w in segs:
                        px0, py0 = int(x0 + dx), int(y0 + dy)
                        px1, py1 = int(x1 + dx), int(y1 + dy)
                        if transparent:
                            cv2.line(canvas, (px0, py0), (px1, py1),
                                     color_bgr + (255,), line_w)
                        else:
                            cv2.line(canvas, (px0, py0), (px1, py1),
                                     color_bgr, line_w)

        return canvas
