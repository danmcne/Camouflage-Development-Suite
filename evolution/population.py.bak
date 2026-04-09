"""
Population manager for the evolution system.
Holds a list of CamoPattern and orchestrates selection, crossover, mutation.
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
    ):
        self.size = size
        self.generator_type = generator_type
        self.colors = colors or []
        self.individuals: list[CamoPattern] = []
        self.generation = 0

    # ── initialisation ────────────────────────────────────────────────────────

    def seed(self):
        """Create an initial random population."""
        gen = get_generator(self.generator_type)
        self.individuals = []
        for _ in range(self.size):
            params = gen.default_params()
            params = gen.mutate(params, strength=0.5)   # randomise from defaults
            p = CamoPattern(
                generator_type=self.generator_type,
                params=params,
                colors=list(self.colors),
            )
            self.individuals.append(p)

    # ── selection ─────────────────────────────────────────────────────────────

    def tournament_select(self, k: int = 3) -> CamoPattern:
        """Return the fittest individual from a random tournament of k."""
        contestants = random.sample(self.individuals, min(k, len(self.individuals)))
        return max(contestants, key=lambda p: p.fitness)

    # ── reproduction ──────────────────────────────────────────────────────────

    def evolve_step(
        self,
        mutation_strength: float = EVOLUTION["mutation_strength"],
        crossover_rate: float = EVOLUTION["crossover_rate"],
        elitism: int = 2,
    ) -> None:
        """
        Produce the next generation in-place.

        Strategy:
          1. Keep top `elitism` individuals unchanged.
          2. Fill remaining slots via tournament selection + crossover + mutation.
        """
        gen = get_generator(self.generator_type)

        # Sort by fitness descending
        self.individuals.sort(key=lambda p: p.fitness, reverse=True)
        next_gen: list[CamoPattern] = self.individuals[:elitism]

        while len(next_gen) < self.size:
            parent_a = self.tournament_select()
            parent_b = self.tournament_select()

            if random.random() < crossover_rate:
                child_params = gen.crossover(parent_a.params, parent_b.params)
            else:
                child_params = copy.deepcopy(parent_a.params)

            child_params = gen.mutate(child_params, mutation_strength)
            child = CamoPattern(
                generator_type=self.generator_type,
                params=child_params,
                colors=list(self.colors),
                generation=self.generation + 1,
            )
            next_gen.append(child)

        self.individuals = next_gen
        self.generation += 1

    # ── interactive mode ──────────────────────────────────────────────────────

    def apply_user_selection(self, kept_indices: list[int]) -> None:
        """
        Called in interactive mode when the user picks survivors.
        Survivors breed to refill the population.
        """
        if not kept_indices:
            return
        survivors = [self.individuals[i] for i in kept_indices if i < len(self.individuals)]
        self.individuals = survivors
        self.evolve_step()

    # ── helpers ───────────────────────────────────────────────────────────────

    def best(self) -> CamoPattern:
        return max(self.individuals, key=lambda p: p.fitness)

    def set_generator(self, name: str):
        self.generator_type = name
        for ind in self.individuals:
            ind.generator_type = name
