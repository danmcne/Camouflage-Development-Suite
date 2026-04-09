"""
Abstract base class for all pattern generators, plus shared utilities.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import random
import numpy as np
import cv2


# ── shared utility ────────────────────────────────────────────────────────────

def apply_transparent_bg(img_bgr: np.ndarray,
                          colors: list[tuple[int, int, int]]) -> np.ndarray:
    """
    Convert a BGR image to BGRA, making pixels that match colors[0] transparent.

    colors[0] is the palette's background colour (RGB tuple).
    In the BGR image those pixels are stored as (B, G, R).
    Returns a BGRA uint8 array.
    """
    
    if not colors:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    r0, g0, b0 = colors[0]          # palette stores RGB
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    # Image stores BGR → background pixel = (b0, g0, r0)
    mask = (
        (bgra[:, :, 0] == int(b0)) &
        (bgra[:, :, 1] == int(g0)) &
        (bgra[:, :, 2] == int(r0))
    )
    bgra[mask, 3] = 0
    return bgra


def toroidal_gaussian(field: np.ndarray,
                       sigma_x: float,
                       sigma_y: float) -> np.ndarray:
    """
    Gaussian blur with toroidal (wrap-around) boundary conditions.
    Works on a 2-D float32 array.
    """

    kw = max(3, int(sigma_x * 6 + 1) | 1)
    kh = max(3, int(sigma_y * 6 + 1) | 1)
    padded  = np.pad(field, ((kh, kh), (kw, kw)), mode="wrap")
    blurred = cv2.GaussianBlur(padded, (kw, kh),
                                sigmaX=sigma_x, sigmaY=sigma_y)
    return blurred[kh:-kh, kw:-kw]


# ── abstract base ─────────────────────────────────────────────────────────────

class BaseGenerator(ABC):
    name: str        = "base"
    description: str = ""

    @abstractmethod
    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],   # (R, G, B) tuples
        params: dict,
    ) -> np.ndarray:
        """
        Generate a pattern image.

        Returns uint8 BGR **or BGRA** NumPy array of shape (height, width, 3|4).
        Return BGRA only when transparent_bg is enabled.
        """

    @abstractmethod
    def get_param_schema(self) -> dict:
        """
        Describe every controllable parameter for the UI auto-builder.

        Schema format:
        {
            "param_name": {
                "default":   <value>,
                "min":       <number>,   # omit for non-numeric
                "max":       <number>,
                "step":      <number>,
                "label":     "Human label",
                "tip":       "Tooltip text",
                "type":      "float"|"int"|"str"|"bool"|"folder"|"choice",
                "options":   ["a","b"],  # only for combo params
                "evolvable": True,       # False = UI-only, not mutated
            }
        }
        """

    def mutate(self, params: dict, strength: float = 0.15) -> dict:
        schema     = self.get_param_schema()
        new_params = copy.deepcopy(params)
        for key, spec in schema.items():
            if not spec.get("evolvable", True):
                continue
            if key not in new_params:
                continue
            val = new_params[key]
            if isinstance(val, bool):
                if random.random() < strength * 0.3:
                    new_params[key] = not val
            elif isinstance(val, float):
                lo, hi = spec.get("min", 0.0), spec.get("max", 1.0)
                new_params[key] = float(
                    np.clip(val + random.gauss(0, strength * (hi - lo)), lo, hi)
                )
            elif isinstance(val, int) and "options" not in spec:
                lo, hi = int(spec.get("min", 0)), int(spec.get("max", 100))
                new_params[key] = int(
                    np.clip(val + int(random.gauss(0, max(1, strength * (hi - lo)))), lo, hi)
                )
            elif isinstance(val, str) and "options" in spec:
                if random.random() < strength * 0.5:
                    new_params[key] = random.choice(spec["options"])
        return new_params

    def crossover(self, params_a: dict, params_b: dict) -> dict:
        schema = self.get_param_schema()
        child  = copy.deepcopy(params_a)
        for key in schema:
            if key not in params_a or key not in params_b:
                continue
            if not schema[key].get("evolvable", True):
                continue
            a, b = params_a[key], params_b[key]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                t = random.random()
                child[key] = type(a)(a * (1 - t) + b * t)
            else:
                child[key] = random.choice([a, b])
        return child

    def default_params(self) -> dict:
        return {k: v["default"] for k, v in self.get_param_schema().items()}
