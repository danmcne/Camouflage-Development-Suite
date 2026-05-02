"""Generator registry."""
from generators.procedural_noise   import ProceduralNoiseGenerator
from generators.blur_sharp         import BlurSharpGenerator
from generators.reaction_diffusion import ReactionDiffusionGenerator
from generators.l_system           import LSystemGenerator
from generators.recursive_fractal  import RecursiveFractalGenerator
from generators.urban_geometric    import UrbanGeometricGenerator
from generators.collage            import CollageGenerator
from generators.dazzle             import DazzleGenerator
from generators.plaid              import PlaidGenerator
from generators.digital_camo       import DigitalCamoGenerator
from generators.argyle             import ArgyleGenerator
from generators.african_pattern    import AfricanPatternGenerator
from generators.japanese_pattern   import JapanesePatternGenerator

REGISTRY: dict[str, type] = {
    "Procedural Noise":   ProceduralNoiseGenerator,
    "Blur-Sharp":         BlurSharpGenerator,
    "Reaction-Diffusion": ReactionDiffusionGenerator,
    "L-System":           LSystemGenerator,
    "Recursive Fractal":  RecursiveFractalGenerator,
    "Urban Geometric":    UrbanGeometricGenerator,
    "Collage":            CollageGenerator,
    "Dazzle":             DazzleGenerator,
    "Plaid":              PlaidGenerator,
    "Digital Camo":       DigitalCamoGenerator,
    "Argyle":             ArgyleGenerator,
    "African Pattern":    AfricanPatternGenerator,
    "Japanese Pattern":   JapanesePatternGenerator,
}

def get_generator(name: str):
    cls = REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown generator: {name!r}. Available: {list(REGISTRY)}")
    return cls()
