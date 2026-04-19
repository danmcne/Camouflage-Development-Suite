"""
CamoPattern – the central data object passed between generator, evolution, and UI.

Layer 2 fields (generator_type2, params2, blend_mode2, opacity2, colors2) are
optional and only populated when a two-layer stack is being used.
colors2 holds the L2 palette as hex strings (same format as colors). It is
separate so the user can evolve with a different palette on each layer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CamoPattern:
    # Layer 1
    generator_type: str = "procedural_noise"
    params: dict         = field(default_factory=dict)
    colors: list[str]    = field(default_factory=list)
    image: Optional[np.ndarray] = None
    fitness: float = 0.0
    generation: int = 0
    uid: str = ""

    # Layer 2 (optional)
    generator_type2: str  = ""
    params2: dict          = field(default_factory=dict)
    blend_mode2: str       = "normal"
    opacity2: float        = 0.5
    colors2: list[str]     = field(default_factory=list)   # L2 palette (hex)

    # PSO / GA internal state
    velocity: dict   = field(default_factory=dict)
    velocity2: dict  = field(default_factory=dict)

    def __post_init__(self):
        if not self.uid:
            import uuid
            self.uid = uuid.uuid4().hex[:8]

    def clone(self) -> "CamoPattern":
        import copy, uuid
        c = copy.deepcopy(self)
        c.uid   = uuid.uuid4().hex[:8]
        c.image = None
        return c

    def to_dict(self) -> dict:
        d = dict(generator_type=self.generator_type, params=self.params,
                 colors=self.colors, fitness=self.fitness,
                 generation=self.generation, uid=self.uid)
        if self.generator_type2:
            d.update(generator_type2=self.generator_type2, params2=self.params2,
                     blend_mode2=self.blend_mode2, opacity2=self.opacity2,
                     colors2=self.colors2)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CamoPattern":
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in valid})
