"""
Generator registry – maps display name → class.
Import from here to get access to all generators.
"""
from generators.procedural_noise   import ProceduralNoiseGenerator
from generators.reaction_diffusion import ReactionDiffusionGenerator
from generators.l_system           import LSystemGenerator
from generators.recursive_fractal  import RecursiveFractalGenerator
from generators.urban_geometric    import UrbanGeometricGenerator
from generators.collage            import CollageGenerator

REGISTRY: dict[str, type] = {
    "Procedural Noise":   ProceduralNoiseGenerator,
    "Reaction-Diffusion": ReactionDiffusionGenerator,
    "L-System":           LSystemGenerator,
    "Recursive Fractal":  RecursiveFractalGenerator,
    "Urban Geometric":    UrbanGeometricGenerator,
    "Collage":            CollageGenerator,
}

def get_generator(name: str):
    """Return an instantiated generator by display name."""
    cls = REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown generator: {name!r}. Available: {list(REGISTRY)}")
    return cls()
