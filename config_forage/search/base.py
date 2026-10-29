"""Searcher protocol, the anytime Trace, and the batched sampling loop shared by BSA and random."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from config_forage.blis.runner import run_batch
from config_forage.bsa.biaser import Biaser
from config_forage.bsa.generator import CategoricalGenerator
from config_forage.bsa.space import Decision
from config_forage.probe import ProbeResult
from config_forage.space import KnobSpace

MAX_RESAMPLE = 10  # tries for a decision not yet scored in this run, then the duplicate is accepted
MAX_STALL = 5  # consecutive batches adding no unique run (space exhausted / P collapsed) end the search


@dataclass(frozen=True)
class TracePoint:
    runs: int  # objective.runs_used right after this evaluation was recorded
    best_score: float | None  # best successful score so far; None until the first success
    best_config: dict | None
    evaluated: int  # evaluations so far, this one included
    decision: list | None = None  # this evaluation's decision (option indices, None = unlabelled)
    score: float | None = None  # this evaluation's score (None = failed); old traces lack both


@dataclass
class Trace:
    points: list[TracePoint]
    method: str
    seed: int
    extra: dict = field(default_factory=dict)  # e.g. "history": BSA per-generation P

    def value_at(self, budget: int) -> float | None:
        """Best score at the last point with runs <= budget (None if none, or none succeeded)."""
        v = None
        for p in self.points:
            if p.runs > budget:
                break
            v = p.best_score
        return v

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> Trace:
        raw = json.loads(s)
        return cls([TracePoint(**p) for p in raw["points"]], raw["method"], raw["seed"], raw.get("extra", {}))

    def top_configs(self, n: int, space: KnobSpace) -> list[tuple[float, dict]]:
        """The n best distinct successful evaluations as (score, config), best first (needs decisions)."""
        best: dict[tuple, float] = {}
        for p in self.points:
            if p.decision is not None and p.score is not None:
                d = tuple(p.decision)
                best[d] = min(p.score, best.get(d, p.score))
        return [(s, space.config(d)) for d, s in sorted(best.items(), key=lambda x: x[1])[:n]]


class Searcher(Protocol):
    name: str

    def search(self, objective, budget: int, seed: int) -> Trace:
        ...


class Recorder:
    """Appends one TracePoint per evaluation, in completion order, and keeps decision -> result."""

    def __init__(self, space: KnobSpace, objective):
        self.space, self.objective = space, objective
        self.points: list[TracePoint] = []
        self.scored: dict[Decision, ProbeResult] = {}
        self.best: tuple[float, Decision] | None = None

    def record(self, d: Decision, r: ProbeResult, runs: int | None = None) -> ProbeResult:
        """runs defaults to objective.runs_used now; callers serialize record(), so it is non-decreasing."""
        runs = self.objective.runs_used if runs is None else runs
        self.scored[d] = r
        if r.score is not None and (self.best is None or r.score < self.best[0]):
            self.best = (r.score, d)
        b = self.best
        self.points.append(TracePoint(runs, None if b is None else b[0],
                                      None if b is None else self.space.config(b[1]), len(self.points) + 1,
                                      [None if i is None else int(i) for i in d], r.score))
        return r


def batch_search(space: KnobSpace, objective, budget: int, seed: int, gen: CategoricalGenerator,
                 biasers: Sequence[Biaser], batch: int, workers: int | None,
                 after: Callable[[list[Decision], list[ProbeResult], Recorder], None]) -> list[TracePoint]:
    """Sample `batch` decisions from gen under biasers, evaluate them in parallel, call `after`;
    repeat until objective.runs_used >= budget (the last batch may overshoot, and the trace shows it).

    Each batch is evaluated in parallel, then recorded in decision order: a point's runs is the
    distinct result keys (ProbeResult.keys) of this search so far, capped at objective.runs_used,
    so a seeded trace does not depend on thread timing.

    Also stops after MAX_STALL consecutive batches that add no unique run: duplicates are free, so
    an exhausted space or a collapsed P would otherwise loop forever."""
    rng = np.random.default_rng(seed)
    rec = Recorder(space, objective)
    stall, base, seen = 0, objective.runs_used, set()

    while objective.runs_used < budget and stall < MAX_STALL:
        before = objective.runs_used
        ds: list[Decision] = []
        for _ in range(batch):
            for _ in range(MAX_RESAMPLE):
                d = gen.sample(rng, biasers)
                if d not in rec.scored and d not in ds:
                    break
            ds.append(d)
        rs = run_batch(objective, ds, workers)
        after_runs = objective.runs_used
        for d, r in zip(ds, rs):
            seen |= r.keys
            rec.record(d, r, min(base + len(seen), after_runs))
        after(ds, rs, rec)
        stall = stall + 1 if after_runs == before else 0
    return rec.points
