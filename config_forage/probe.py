"""Concurrency probe to C* (max SLO-passing concurrency) and the scalar objective.

Score = -(tokens_per_sec / gpus) at c_use = max(1, floor(h·C*)); None for a failed config (lower is better)."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

from config_forage.bsa.space import Decision
from config_forage.space import KnobSpace, Scenario, Slo

__all__ = ["Slo", "ProbeResult", "ladder", "probe", "Objective"]


@dataclass(frozen=True)
class ProbeResult:
    score: float | None
    c_star: int | None
    c_use: int | None
    tps_per_gpu: float | None
    metrics_at_use: dict
    diagnostics: dict
    runs: int  # distinct cache keys this probe touched
    keys: frozenset[str] = field(default_factory=frozenset)


def ladder(gpu_budget: int) -> list[int]:
    """Probe ladder (upstream concurrency_ramp): 1..32, then doubling to max(32, 2·budget)."""
    ramp, c, top = [1, 2, 4, 8, 16, 24, 32], 64, max(32, gpu_budget * 2)
    while c <= top:
        ramp.append(c)
        c *= 2
    return ramp


def probe(runner, config: dict, scenario: Scenario, seed: int | None = None) -> ProbeResult:
    slo, recs = scenario.slo, {}

    def run(c: int) -> dict:
        if c not in recs:
            recs[c] = runner.run(config, c, seed)
        return recs[c]

    def ok(c: int) -> bool:
        r = run(c)
        return r.get("error") is None and slo.passes(r.get("metrics") or {})

    def result(score=None, c_star=None, c_use=None, tps=None, m=None, diag=None) -> ProbeResult:
        keys = frozenset(r["cache_key"] for r in recs.values())
        return ProbeResult(score, c_star, c_use, tps, m or {}, diag or {}, len(keys), keys)

    lo, hi = None, None  # last pass, first fail
    for c in ladder(scenario.gpu_budget):
        if not ok(c):
            hi = c
            break
        lo = c
    if lo is None:
        return result()
    while hi is not None and hi - lo > 1:  # a run error counts as an SLO failure
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    c_star = lo

    m_star = recs[c_star].get("metrics") or {}
    pre = int(m_star.get("preemption_count") or 0)
    diag = {"preemptions": pre, "kv_starved": pre > 0, "ttft_bound": False, "binding": None}
    if hi is not None:
        f = recs[hi]
        fm = f.get("metrics") or {}
        itl_ok = Slo(slo.itl_p99_ms).passes(fm)
        if f.get("error") is not None:
            diag["binding"] = "error"
        else:
            diag["binding"] = "itl" if not itl_ok else "ttft"
        diag["ttft_bound"] = (f.get("error") is None and slo.ttft_p99_ms is not None and itl_ok
                              and not slo.passes(fm))

    c_use = max(1, math.floor(scenario.headroom * c_star))
    if not ok(c_use):  # possible only if non-monotone: fall back to the best measured pass below
        passing = [c for c in recs if c <= c_use and ok(c)]
        if not passing:
            return result(c_star=c_star, diag=diag)
        c_use = max(passing)
    rec = recs[c_use]
    m = rec.get("metrics") or {}
    gpus = (rec.get("deployment") or {}).get("gpus")
    tps = m.get("tokens_per_sec")
    if not gpus or tps is None:
        return result(c_star=c_star, c_use=c_use, m=m, diag=diag)
    return result(-(tps / gpus), c_star, c_use, tps / gpus, m, diag)


class Objective:
    """Decision -> ProbeResult; runs_used = distinct cache keys touched (hits included), thread-safe."""

    def __init__(self, space: KnobSpace, runner):
        self.space, self.runner = space, runner
        self._keys: set[str] = set()
        self._lock = threading.Lock()
        self.evaluations = 0

    @property
    def runs_used(self) -> int:
        with self._lock:
            return len(self._keys)

    def __call__(self, d: Decision) -> ProbeResult:
        p = probe(self.runner, self.space.config(d), self.runner.scenario)
        with self._lock:
            self._keys |= p.keys
            self.evaluations += 1
        return p
