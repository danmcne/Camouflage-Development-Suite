"""
Population manager for the evolution system.

Supports single-generator and two-generator (layered) populations.
When use_layer2=True each individual carries params for both generators;
mutation and crossover are applied independently to each layer's params.
"""
from __future__ import annotations
import random
import copy
from core.pattern import CamoPattern
from generators import get_generator
from config.defaults import EVOLUTION


class Population:
    def __init__(
        self,
        size: int = EVOLUTION["population_size"],
        generator_type: str = "Procedural Noise",
        colors: list[str] | None = None,
        base_params: dict | None = None,
        # Layer 2 (optional)
        use_layer2: bool = False,
        generator_type2: str = "",
        base_params2: dict | None = None,
        blend_mode2: str = "normal",
        opacity2: float = 0.5,
    ):
        self.size           = size
        self.generator_type = generator_type
        self.colors         = colors or []
        self.base_params    = base_params or {}
        self.use_layer2     = use_layer2
        self.generator_type2= generator_type2
        self.base_params2   = base_params2 or {}
        self.blend_mode2    = blend_mode2
        self.opacity2       = opacity2
        self.individuals: list[CamoPattern] = []
        self.generation     = 0

    # ── initialisation ────────────────────────────────────────────────────────

    def seed(self):
        gen    = get_generator(self.generator_type)
        schema = gen.get_param_schema()

        gen2   = get_generator(self.generator_type2) if self.use_layer2 and self.generator_type2 else None
        schema2= gen2.get_param_schema() if gen2 else {}

        self.individuals = []
        for _ in range(self.size):
            # Layer 1
            params = gen.default_params()
            params.update(self.base_params)
            params = gen.mutate(params, strength=0.5)
            for k, v in self.base_params.items():
                if not schema.get(k, {}).get("evolvable", True):
                    params[k] = v

            p = CamoPattern(
                generator_type=self.generator_type,
                params=params,
                colors=list(self.colors),
            )

            # Layer 2
            if gen2 is not None:
                params2 = gen2.default_params()
                params2.update(self.base_params2)
                params2 = gen2.mutate(params2, strength=0.5)
                for k, v in self.base_params2.items():
                    if not schema2.get(k, {}).get("evolvable", True):
                        params2[k] = v
                p.generator_type2 = self.generator_type2
                p.params2         = params2
                p.blend_mode2     = self.blend_mode2
                p.opacity2        = self.opacity2

            self.individuals.append(p)

    # ── selection ─────────────────────────────────────────────────────────────

    def tournament_select(self, k: int = 3) -> CamoPattern:
        contestants = random.sample(self.individuals, min(k, len(self.individuals)))
        return max(contestants, key=lambda p: p.fitness)

    # ── reproduction ──────────────────────────────────────────────────────────

    def evolve_step(
        self,
        mutation_strength: float = EVOLUTION["mutation_strength"],
        crossover_rate:    float = EVOLUTION["crossover_rate"],
        elitism:           int   = 2,
    ) -> None:
        gen  = get_generator(self.generator_type)
        gen2 = (get_generator(self.generator_type2)
                if self.use_layer2 and self.generator_type2 else None)

        self.individuals.sort(key=lambda p: p.fitness, reverse=True)
        next_gen: list[CamoPattern] = self.individuals[:elitism]

        while len(next_gen) < self.size:
            pa = self.tournament_select()
            pb = self.tournament_select()

            # Layer 1
            if random.random() < crossover_rate:
                child_params = gen.crossover(pa.params, pb.params)
            else:
                child_params = copy.deepcopy(pa.params)
            child_params = gen.mutate(child_params, mutation_strength)

            child = CamoPattern(
                generator_type=self.generator_type,
                params=child_params,
                colors=list(self.colors),
                generation=self.generation + 1,
            )

            # Layer 2
            if gen2 is not None:
                if random.random() < crossover_rate:
                    child_params2 = gen2.crossover(pa.params2, pb.params2)
                else:
                    child_params2 = copy.deepcopy(pa.params2)
                child_params2 = gen2.mutate(child_params2, mutation_strength)
                child.generator_type2 = self.generator_type2
                child.params2         = child_params2
                child.blend_mode2     = self.blend_mode2
                child.opacity2        = self.opacity2

            next_gen.append(child)

        self.individuals = next_gen
        self.generation += 1

    # ── interactive mode ──────────────────────────────────────────────────────

    def apply_user_selection(self, kept_indices: list[int]) -> None:
        if not kept_indices:
            return
        survivors = [self.individuals[i] for i in kept_indices
                     if i < len(self.individuals)]
        self.individuals = survivors
        self.evolve_step()

    # ── helpers ───────────────────────────────────────────────────────────────

    def best(self) -> CamoPattern:
        return max(self.individuals, key=lambda p: p.fitness)

    def set_generator(self, name: str):
        self.generator_type = name
        for ind in self.individuals:
            ind.generator_type = name
