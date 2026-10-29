"""Test doubles shared by the searcher tests: a synthetic 4-axis space and objective.

SYNTHETIC_SCORE (lower is better), on knobs a, b, c, d each in 0..5 (default 0):

    score = 2(a + b) - a*b + (c - 3)^2 + (d - 2)^2

The unique optimum is (a, b, c, d) = (5, 5, 3, 2) with score -5. The a*b interaction means a and
b pay off only together: with the other at 0, raising a or b alone costs 2 per step, so from the
default a coordinatewise search settles at (0, 0, 3, 2), score 0, and misses the optimum (Task 7
relies on this). A one-cell bonus (a == a* and b == b*) was tried first: random hits a 1/36 region
too often at 150 runs to separate from BSA; the bilinear term gives BSA a gradient to follow."""

from __future__ import annotations

import threading

from config_forage.probe import ProbeResult
from config_forage.space import Knob, KnobSpace, Scenario

OPTIMUM = {"a": 5, "b": 5, "c": 3, "d": 2}
OPT_SCORE = -5.0


def synthetic_score(cfg: dict) -> float:
    a, b = cfg["a"], cfg["b"]
    return float(2 * (a + b) - a * b + (cfg["c"] - 3) ** 2 + (cfg["d"] - 2) ** 2)


def synthetic_space() -> KnobSpace:
    """6^4 = 1296 combinations, no topology knobs, so every decision is feasible."""
    sc = Scenario("synthetic", "h100", "chatbot", 8, "0.29.0")
    return KnobSpace(sc, tuple(Knob(n, tuple(range(6))) for n in "abcd"))


def empty_space() -> KnobSpace:
    """gpu_budget 3 with tp in {2, 4}: no feasible decision."""
    sc = Scenario("dense", "h100", "chatbot", 3, "0.29.0")
    return KnobSpace(sc, (Knob("tp", (2, 4)), Knob("max_num_seqs", (64, 128))))


class SyntheticObjective:
    """Objective look-alike: score_fn(config) -> float | None; each distinct decision costs
    `runs_per_eval` runs once, a repeat costs 0 (as the cache). Thread-safe."""

    def __init__(self, space: KnobSpace, score_fn=synthetic_score, runs_per_eval: int = 1):
        self.space, self.score_fn, self.runs_per_eval = space, score_fn, runs_per_eval
        self._seen: set = set()
        self._lock = threading.Lock()
        self.evaluations = 0
        self.calls: list = []

    @property
    def runs_used(self) -> int:
        with self._lock:
            return len(self._seen) * self.runs_per_eval

    def __call__(self, d) -> ProbeResult:
        s = self.score_fn(self.space.config(d))
        keys = frozenset(f"{d}:{i}" for i in range(self.runs_per_eval))
        with self._lock:
            self._seen.add(d)
            self.evaluations += 1
            self.calls.append(d)
        return ProbeResult(s, None, None, None if s is None else -s, {}, {}, self.runs_per_eval, keys)
