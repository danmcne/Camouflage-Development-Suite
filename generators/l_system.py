"""
L-System Generator – multiple trees, per-tree variation, toroidal turtle.

Each tree shares the same axiom/rules but gets its own:
  • starting position  (random across canvas)
  • starting direction (random full rotation)
  • angle             (base_angle ± angle_var)
  • step size         (uniform in [step_min, step_max])
  • line width        (uniform int in [width_min, width_max])
  • colour            (random from palette if color_per_tree, else cycling)

Turtle positions are wrapped modulo (width, height) for seamless tiling.
Lines that cross a boundary are split at the crossing point so no long
diagonal artefacts appear.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from generators.base import BaseGenerator
from config.defaults import GENERATORS


def _expand(axiom: str, rules: dict, iterations: int) -> str:
    current = axiom
    for _ in range(iterations):
        current = "".join(rules.get(ch, ch) for ch in current)
    return current


def _parse_rules(rules_str: str) -> dict:
    rules = {}
    for rule in rules_str.split(";"):
        rule = rule.strip()
        if "->" in rule:
            lhs, rhs = rule.split("->", 1)
            rules[lhs.strip()] = rhs.strip()
    return rules


def _draw_toroidal_line(canvas, x0, y0, x1, y1, color, width, W, H):
    """
    Draw a line that wraps at canvas boundaries.
    Splits the segment at each crossing and draws each piece separately.
    """
    dx = x1 - x0
    dy = y1 - y0
    steps = max(int(math.hypot(dx, dy) / max(W, H) * 100) + 1, 1)

    prev_x = x0 % W
    prev_y = y0 % H

    for i in range(1, steps + 1):
        t = i / steps
        raw_x = x0 + dx * t
        raw_y = y0 + dy * t
        cur_x = raw_x % W
        cur_y = raw_y % H

        # If wrapping occurred (jump), don't draw this segment
        if abs(cur_x - prev_x) < W / 2 and abs(cur_y - prev_y) < H / 2:
            cv2.line(canvas,
                     (int(prev_x), int(prev_y)),
                     (int(cur_x),  int(cur_y)),
                     color, width)

        prev_x, prev_y = cur_x, cur_y


class LSystemGenerator(BaseGenerator):
    name = "L-System"
    description = (
        "Multiple turtle-based L-System trees scattered across the canvas. "
        "Each tree has independent angle, step size, width and colour. "
        "Toroidal wrapping for seamless patterns."
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
        seed        = int(params.get("seed", 42))

        rng = np.random.default_rng(seed)
        rules = _parse_rules(rules_str)

        n_colors = max(1, len(colors))

        # ── Expand L-system once (shared across all trees) ────────────────────
        sentence = _expand(axiom, rules, iterations)

        # ── Background = first colour ─────────────────────────────────────────
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if colors:
            r0, g0, b0 = colors[0]
            canvas[:] = (int(b0), int(g0), int(r0))

        # ── Draw each tree ────────────────────────────────────────────────────
        for tree_idx in range(num_trees):
            # Per-tree parameters
            angle_deg  = base_angle + rng.uniform(-angle_var, angle_var)
            angle_rad  = math.radians(angle_deg)
            step       = rng.uniform(step_min, step_max)
            lw         = int(rng.integers(width_min, width_max + 1))

            if color_per:
                ci = int(rng.integers(0, n_colors))
            else:
                ci = tree_idx % n_colors
            r, g, b = colors[ci]
            color_bgr = (int(b), int(g), int(r))

            # Random start position, random initial heading
            x   = rng.uniform(0, width)
            y   = rng.uniform(0, height)
            direction = rng.uniform(0, 2 * math.pi)

            stack = []

            for ch in sentence:
                if ch in ("F", "G"):
                    nx = x + step * math.cos(direction)
                    ny = y + step * math.sin(direction)
                    _draw_toroidal_line(canvas,
                                        x, y, nx, ny,
                                        color_bgr, lw, width, height)
                    x, y = nx, ny

                elif ch == "+":
                    direction += angle_rad

                elif ch == "-":
                    direction -= angle_rad

                elif ch == "[":
                    stack.append((x, y, direction))

                elif ch == "]":
                    if stack:
                        x, y, direction = stack.pop()

                elif ch == "f":
                    x += step * math.cos(direction)
                    y += step * math.sin(direction)

                elif ch == "|":
                    direction += math.pi

        return canvas
