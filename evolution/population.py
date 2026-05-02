"""
Population manager. Supports single-generator and two-layer populations.
Layer 2 has its own palette (colors2) so each layer can use a different set.
"""
from __future__ import annotations
import random, copy
from core.pattern import CamoPattern
from generators import get_generator
from config.defaults import EVOLUTION


class Population:
    def __init__(self,
                 size: int = EVOLUTION["population_size"],
                 generator_type: str = "Procedural Noise",
                 colors: list[str] | None = None,
                 base_params: dict | None = None,
                 use_layer2: bool = False,
                 generator_type2: str = "",
                 base_params2: dict | None = None,
                 blend_mode2: str = "normal",
                 opacity2: float = 0.5,
                 colors2: list[str] | None = None,
                 locked_params: set | None = None,
                 locked_params2: set | None = None):
        self.size            = size
        self.generator_type  = generator_type
        self.colors          = colors or []
        self.base_params     = base_params or {}
        self.use_layer2      = use_layer2
        self.generator_type2 = generator_type2
        self.base_params2    = base_params2 or {}
        self.blend_mode2     = blend_mode2
        self.opacity2        = opacity2
        self.colors2         = colors2 or []
        self.locked_params   = set(locked_params)  if locked_params  else set()
        self.locked_params2  = set(locked_params2) if locked_params2 else set()
        self.individuals: list[CamoPattern] = []
        self.generation      = 0

    def seed(self):
        gen    = get_generator(self.generator_type)
        schema = gen.get_param_schema()
        gen2   = get_generator(self.generator_type2) if self.use_layer2 and self.generator_type2 else None
        schema2= gen2.get_param_schema() if gen2 else {}
        self.individuals = []
        for _ in range(self.size):
            params = gen.default_params()
            params.update(self.base_params)
            params = gen.mutate(params, strength=0.5, locked=self.locked_params)
            # Always restore locked and non-evolvable keys from base_params
            for k, v in self.base_params.items():
                if not schema.get(k, {}).get("evolvable", True) or k in self.locked_params:
                    params[k] = v
            p = CamoPattern(generator_type=self.generator_type,
                            params=params, colors=list(self.colors))
            if gen2 is not None:
                params2 = gen2.default_params()
                params2.update(self.base_params2)
                params2 = gen2.mutate(params2, strength=0.5, locked=self.locked_params2)
                for k, v in self.base_params2.items():
                    if not schema2.get(k, {}).get("evolvable", True) or k in self.locked_params2:
                        params2[k] = v
                p.generator_type2 = self.generator_type2
                p.params2         = params2
                p.blend_mode2     = self.blend_mode2
                p.opacity2        = self.opacity2
                p.colors2         = list(self.colors2)
            self.individuals.append(p)

    def tournament_select(self, k: int = 3) -> CamoPattern:
        contestants = random.sample(self.individuals, min(k, len(self.individuals)))
        return max(contestants, key=lambda p: p.fitness)

    def evolve_step(self,
                    mutation_strength: float = EVOLUTION["mutation_strength"],
                    crossover_rate:    float = EVOLUTION["crossover_rate"],
                    elitism:           int   = 2) -> None:
        gen  = get_generator(self.generator_type)
        gen2 = (get_generator(self.generator_type2)
                if self.use_layer2 and self.generator_type2 else None)
        self.individuals.sort(key=lambda p: p.fitness, reverse=True)
        next_gen = self.individuals[:elitism]
        while len(next_gen) < self.size:
            pa = self.tournament_select(); pb = self.tournament_select()
            child_params = (gen.crossover(pa.params, pb.params, locked=self.locked_params)
                            if random.random() < crossover_rate
                            else copy.deepcopy(pa.params))
            child_params = gen.mutate(child_params, mutation_strength, locked=self.locked_params)
            # Restore locked params from best parent
            for k in self.locked_params:
                if k in pa.params:
                    child_params[k] = pa.params[k]
            child = CamoPattern(generator_type=self.generator_type,
                                params=child_params, colors=list(self.colors),
                                generation=self.generation + 1)
            if gen2 is not None:
                child_params2 = (gen2.crossover(pa.params2, pb.params2,
                                                   locked=self.locked_params2)
                                 if random.random() < crossover_rate
                                 else copy.deepcopy(pa.params2))
                child_params2 = gen2.mutate(child_params2, mutation_strength,
                                            locked=self.locked_params2)
                # Restore locked L2 params from best parent
                for k in self.locked_params2:
                    if k in pa.params2:
                        child_params2[k] = pa.params2[k]
                child.generator_type2 = self.generator_type2
                child.params2         = child_params2
                child.blend_mode2     = self.blend_mode2
                child.opacity2        = self.opacity2
                child.colors2         = list(self.colors2)
            next_gen.append(child)
        self.individuals = next_gen
        self.generation += 1

    def apply_user_selection(self, kept_indices: list[int]) -> None:
        if not kept_indices: return
        self.individuals = [self.individuals[i] for i in kept_indices
                            if i < len(self.individuals)]
        self.evolve_step()

    def best(self) -> CamoPattern:
        return max(self.individuals, key=lambda p: p.fitness)

    def set_generator(self, name: str):
        self.generator_type = name
        for ind in self.individuals:
            ind.generator_type = name
