"""BSA searcher: a CategoricalGenerator under explicit biasers, updated toward each batch's elites."""

from __future__ import annotations

from collections.abc import Sequence

from config_forage.bsa.biaser import Biaser
from config_forage.bsa.generator import CategoricalGenerator, Scheme
from config_forage.feasibility import enumerate_feasible
from config_forage.search.base import Trace, batch_search
from config_forage.space import KnobSpace


class BsaSearcher:
    """The feasibility biaser is not implied: pass it in `biasers`, so a uniform arm stays uniform.

    After each generation, every biaser with an `observe` method gets [(decision, ProbeResult), ...]
    of that generation. `history` holds {gen, runs, entropy, P, ess} per generation (also in
    trace.extra["history"])."""

    def __init__(self, space: KnobSpace, biasers: Sequence[Biaser] = (), batch: int = 16, alpha: float = 0.7,
                 rho: float = 0.3, eps: float = 0.02, scheme: Scheme = "hard", workers: int | None = None,
                 name: str = "bsa"):
        self.space, self.biasers, self.batch, self.workers, self.name = space, tuple(biasers), batch, workers, name
        self.alpha, self.rho, self.eps, self.scheme = alpha, rho, eps, scheme
        self.history: list[dict] = []

    def search(self, objective, budget: int, seed: int) -> Trace:
        if next(enumerate_feasible(self.space), None) is None:
            raise ValueError(f"no feasible decision in the space for scenario {self.space.scenario}")
        gen = CategoricalGenerator(self.space.decision_space, self.alpha, self.rho, self.eps, self.scheme)
        self.history = []

        def after(ds, rs, rec):
            gen.update(ds, [r.score for r in rs])
            for b in self.biasers:
                if hasattr(b, "observe"):
                    b.observe(list(zip(ds, rs)))
            self.history.append({"gen": len(self.history), "runs": objective.runs_used, "entropy": gen.entropy(),
                                 "P": gen.snapshot(), "ess": gen.ess})

        points = batch_search(self.space, objective, budget, seed, gen, self.biasers, self.batch, self.workers, after)
        return Trace(points, self.name, seed, {"history": self.history})
