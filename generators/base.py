"""
Abstract base class for all pattern generators.

Every generator must implement:
  generate()        → np.ndarray (H×W×3 uint8 BGR)
  get_param_schema()→ dict  (used by UI to auto-build param panels)
  mutate()          → dict  (used by evolution system)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import random
import numpy as np


class BaseGenerator(ABC):
    """Contract every generator must satisfy."""

    name: str = "base"          # display name
    description: str = ""       # shown in UI tooltip

    @abstractmethod
    def generate(
        self,
        width: int,
        height: int,
        colors: list[tuple[int, int, int]],   # list of (R,G,B) tuples
        params: dict,
    ) -> np.ndarray:
        """
        Generate a pattern image.

        Returns:
            uint8 BGR NumPy array of shape (height, width, 3)
        """

    @abstractmethod
    def get_param_schema(self) -> dict:
        """
        Describe every controllable parameter for the UI auto-builder.

        Schema format (each key is a param name):
        {
            "param_name": {
                "default": <value>,
                "min":     <number>,   # omit for non-numeric
                "max":     <number>,
                "step":    <number>,
                "label":   "Human label",
                "tip":     "Tooltip text",
                "type":    "float"|"int"|"str"|"choice",   # inferred if absent
                "options": ["a","b"],  # only for type=="choice"
                "evolvable": True,     # default True; False = UI only
            }
        }
        """

    def mutate(self, params: dict, strength: float = 0.15) -> dict:
        """
        Return a mutated copy of params for the evolution system.

        Default implementation applies Gaussian noise to numeric params
        within their [min, max] bounds.  Override for custom behaviour.
        """
        schema = self.get_param_schema()
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
                rng = hi - lo
                delta = random.gauss(0, strength * rng)
                new_params[key] = float(np.clip(val + delta, lo, hi))

            elif isinstance(val, int) and "options" not in spec:
                lo, hi = int(spec.get("min", 0)), int(spec.get("max", 100))
                delta = int(random.gauss(0, max(1, strength * (hi - lo))))
                new_params[key] = int(np.clip(val + delta, lo, hi))

            elif isinstance(val, str) and "options" in spec:
                if random.random() < strength * 0.5:
                    new_params[key] = random.choice(spec["options"])

        return new_params

    def crossover(self, params_a: dict, params_b: dict) -> dict:
        """
        Uniform crossover: for each param randomly pick from a or b.
        Numeric params may also be linearly interpolated.
        """
        schema = self.get_param_schema()
        child = copy.deepcopy(params_a)

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
        """Return a params dict filled with schema defaults."""
        return {k: v["default"] for k, v in self.get_param_schema().items()}
