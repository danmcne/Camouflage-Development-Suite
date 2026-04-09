"""Generator registry."""
from generators.procedural_noise   import ProceduralNoiseGenerator
from generators.blur_sharp         import BlurSharpGenerator
from generators.reaction_diffusion import ReactionDiffusionGenerator
from generators.l_system           import LSystemGenerator
from generators.recursive_fractal  import RecursiveFractalGenerator
from generators.urban_geometric    import UrbanGeometricGenerator
from generators.collage            import CollageGenerator

REGISTRY: dict[str, type] = {
    "Procedural Noise":   ProceduralNoiseGenerator,
    "Blur-Sharp":         BlurSharpGenerator,
    "Reaction-Diffusion": ReactionDiffusionGenerator,
    "L-System":           LSystemGenerator,
    "Recursive Fractal":  RecursiveFractalGenerator,
    "Urban Geometric":    UrbanGeometricGenerator,
    "Collage":            CollageGenerator,
}

def get_generator(name: str):
    cls = REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown generator: {name!r}. Available: {list(REGISTRY)}")
    return cls()
