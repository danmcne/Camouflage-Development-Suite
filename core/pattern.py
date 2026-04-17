"""
CamoPattern – the central data object passed between generator, evolution, and UI.

Supports an optional second generator layer (generator_type2 / params2 / blend_mode2 / opacity2)
so that the evolution system can jointly evolve two-layer stacks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CamoPattern:
    """Holds everything needed to reproduce and display one camouflage pattern."""

    # Layer 1 (always present)
    generator_type: str = "procedural_noise"
    params: dict = field(default_factory=dict)
    colors: list[str] = field(default_factory=list)   # list of "#RRGGBB" hex strings
    image: Optional[np.ndarray] = None                # H×W×3 uint8 BGR (OpenCV native)
    fitness: float = 0.0
    generation: int = 0
    uid: str = ""

    # Layer 2 (optional – used when evolving a 2-generator stack)
    generator_type2: str = ""
    params2: dict = field(default_factory=dict)
    blend_mode2: str = "normal"
    opacity2: float = 0.5

    # PSO / GA internal state
    velocity: dict = field(default_factory=dict)
    velocity2: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.uid:
            import uuid
            self.uid = uuid.uuid4().hex[:8]

    def clone(self) -> "CamoPattern":
        import copy
        c = copy.deepcopy(self)
        import uuid
        c.uid = uuid.uuid4().hex[:8]
        c.image = None
        return c

    def to_dict(self) -> dict:
        d = {
            "generator_type":  self.generator_type,
            "params":          self.params,
            "colors":          self.colors,
            "fitness":         self.fitness,
            "generation":      self.generation,
            "uid":             self.uid,
        }
        if self.generator_type2:
            d.update({
                "generator_type2": self.generator_type2,
                "params2":         self.params2,
                "blend_mode2":     self.blend_mode2,
                "opacity2":        self.opacity2,
            })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CamoPattern":
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})
