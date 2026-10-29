"""Random search baseline."""

from __future__ import annotations

from config_forage.bsa.generator import CategoricalGenerator
from config_forage.feasibility import FeasibilityBiaser, feasible_topologies
from config_forage.search.base import Trace, batch_search
from config_forage.space import KnobSpace


class RandomSearcher:
    """Samples from a uniform generator that is never updated. With feasible_only, through a
    FeasibilityBiaser: uniform per axis subject to feasibility (each axis uniform over the options
    feasible given the earlier ones), not uniform over the joint feasible set. Either way the space's
    applicability_biaser applies (an inapplicable conditional knob is never labelled)."""

    def __init__(self, space: KnobSpace, feasible_only: bool = True, batch: int = 16, workers: int | None = None,
                 name: str = "random"):
        self.space, self.feasible_only, self.batch, self.workers, self.name = space, feasible_only, batch, workers, name

    def search(self, objective, budget: int, seed: int) -> Trace:
        if self.feasible_only and not feasible_topologies(self.space):
            raise ValueError(f"no feasible decision in the space for scenario {self.space.scenario}")
        gen = CategoricalGenerator(self.space.decision_space)
        applic = self.space.applicability_biaser()
        biasers = (FeasibilityBiaser(self.space),) if self.feasible_only else ()
        biasers = (applic, *biasers) if applic is not None else biasers
        points = batch_search(self.space, objective, budget, seed, gen, biasers, self.batch, self.workers,
                              lambda ds, rs, rec: None)
        return Trace(points, self.name, seed)
