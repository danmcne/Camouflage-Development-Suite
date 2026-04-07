"""
CamoPattern – the central data object passed between generator, evolution, and UI.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CamoPattern:
    """Holds everything needed to reproduce and display one camouflage pattern."""

    generator_type: str = "procedural_noise"
    params: dict = field(default_factory=dict)
    colors: list[str] = field(default_factory=list)   # list of "#RRGGBB" hex strings
    image: Optional[np.ndarray] = None                # H×W×3 uint8 BGR (OpenCV native)
    fitness: float = 0.0
    generation: int = 0
    uid: str = ""                                     # set on creation for deduplication

    # PSO / GA internal state (ignored by UI)
    velocity: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.uid:
            import uuid
            self.uid = uuid.uuid4().hex[:8]

    def clone(self) -> "CamoPattern":
        import copy
        c = copy.deepcopy(self)
        import uuid
        c.uid = uuid.uuid4().hex[:8]
        c.image = None          # don't copy heavy pixel data
        return c

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict (image excluded)."""
        return {
            "generator_type": self.generator_type,
            "params": self.params,
            "colors": self.colors,
            "fitness": self.fitness,
            "generation": self.generation,
            "uid": self.uid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CamoPattern":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
