"""Searcher protocol, the anytime Trace, and the batched sampling loop shared by BSA and random."""

from __future__ import annotations

import json
import threading
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

    def record(self, d: Decision, r: ProbeResult) -> ProbeResult:
        runs = self.objective.runs_used  # callers serialize record(), so runs is non-decreasing
        self.scored[d] = r
        if r.score is not None and (self.best is None or r.score < self.best[0]):
            self.best = (r.score, d)
        b = self.best
        self.points.append(TracePoint(runs, None if b is None else b[0],
                                      None if b is None else self.space.config(b[1]), len(self.points) + 1))
        return r


def batch_search(space: KnobSpace, objective, budget: int, seed: int, gen: CategoricalGenerator,
                 biasers: Sequence[Biaser], batch: int, workers: int | None,
                 after: Callable[[list[Decision], list[ProbeResult], Recorder], None]) -> list[TracePoint]:
    """Sample `batch` decisions from gen under biasers, evaluate them in parallel, call `after`;
    repeat until objective.runs_used >= budget (the last batch may overshoot, and the trace shows it).

    Also stops after MAX_STALL consecutive batches that add no unique run: duplicates are free, so
    an exhausted space or a collapsed P would otherwise loop forever."""
    rng = np.random.default_rng(seed)
    rec = Recorder(space, objective)
    stall, lock = 0, threading.Lock()

    def evaluate(d: Decision) -> ProbeResult:
        r = objective(d)
        with lock:  # runs_used is read right after this evaluation, before any other is recorded
            return rec.record(d, r)

    while objective.runs_used < budget and stall < MAX_STALL:
        before = objective.runs_used
        ds: list[Decision] = []
        for _ in range(batch):
            for _ in range(MAX_RESAMPLE):
                d = gen.sample(rng, biasers)
                if d not in rec.scored and d not in ds:
                    break
            ds.append(d)
        rs = run_batch(evaluate, ds, workers)
        after(ds, rs, rec)
        stall = stall + 1 if objective.runs_used == before else 0
    return rec.points
