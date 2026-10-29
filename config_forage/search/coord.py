"""Coordinate baselines: the upstream incumbent heuristic and plain coordinate ascent.

IncumbentSearcher follows blis-config-search run_search.py: phase 1 probes every feasible topology
(tp, dp, enable_expert_parallel) with other knobs at defaults; phase 2 takes the best `refine_top`
topologies and probes one-knob-at-a-time variants. Deviation from upstream: variants try EVERY value
of each non-topology knob, not the single KNOB_TIERS value. Deterministic; seed is ignored.

trace.extra["stop"] is "budget", "converged" (coord-ascent) or "done"."""

from __future__ import annotations

import numpy as np

from config_forage.feasibility import TOPOLOGY, feasible_topologies, is_feasible
from config_forage.search.base import Recorder, Trace
from config_forage.space import KnobSpace


def _normalize(space: KnobSpace, cfg: dict) -> dict:
    """Drop inapplicable conditional knobs; give newly applicable ones their default."""
    out: dict = {}
    for k in space.knobs:
        if k.requires is None or out.get(k.requires[0]) in k.requires[1]:
            out[k.name] = cfg.get(k.name, k.default)
    return out


class _Eval:
    """Budget-checked, memoized one-at-a-time evaluation."""

    def __init__(self, space, objective, budget):
        self.space, self.obj, self.budget = space, objective, budget
        self.rec, self.memo, self.out_of_budget = Recorder(space, objective), {}, False

    def __call__(self, cfg):
        d = self.space.decision(cfg)
        if d in self.memo:
            return self.memo[d]
        if self.obj.runs_used >= self.budget:
            self.out_of_budget = True
            return None
        self.memo[d] = self.rec.record(d, self.obj(d)).score
        return self.memo[d]


class IncumbentSearcher:
    def __init__(self, space: KnobSpace, refine_top: int = 3):
        self.space, self.refine_top, self.name = space, refine_top, "incumbent"

    def search(self, objective, budget: int, seed: int) -> Trace:
        sp = self.space
        topos = feasible_topologies(sp)  # topology prefixes only: never walks the full space
        ev = _Eval(sp, objective, budget)
        scored = []
        for t in topos:
            s = ev(_normalize(sp, {**{k.name: k.default for k in sp.knobs}, **t}))
            if ev.out_of_budget:
                return Trace(ev.rec.points, self.name, seed, {"stop": "budget"})
            scored.append((s, len(scored), t))
        ok = sorted((x for x in scored if x[0] is not None), key=lambda x: (x[0], x[1]))
        for _, _, t in ok[:self.refine_top]:
            base = _normalize(sp, {**{k.name: k.default for k in sp.knobs}, **t})
            for k in sp.knobs:
                if k.name in TOPOLOGY or k.name not in base:
                    continue
                for v in k.values:
                    if v == base[k.name]:
                        continue
                    ev(_normalize(sp, {**base, k.name: v}))
                    if ev.out_of_budget:
                        return Trace(ev.rec.points, self.name, seed, {"stop": "budget"})
        return Trace(ev.rec.points, self.name, seed, {"stop": "done"})


class CoordAscentSearcher:
    def __init__(self, space: KnobSpace, start: dict | None = None, max_sweeps: int = 5):
        self.space, self.start, self.max_sweeps, self.name = space, start, max_sweeps, "coord-ascent"

    def search(self, objective, budget: int, seed: int) -> Trace:
        sp = self.space
        cur = _normalize(sp, self.start if self.start is not None else sp.default_config())
        order = list(range(len(sp.knobs)))
        if seed is not None:
            order = [int(i) for i in np.random.default_rng(seed).permutation(len(order))]
        ev = _Eval(sp, objective, budget)

        def stop(why):
            return Trace(ev.rec.points, self.name, seed, {"stop": why})

        best = ev(cur)
        if ev.out_of_budget:
            return stop("budget")
        for _ in range(self.max_sweeps):
            moved = False
            for i in order:
                k = sp.knobs[i]
                if k.name not in cur:
                    continue
                for v in k.values:
                    if v == cur[k.name]:
                        continue
                    cand = _normalize(sp, {**cur, k.name: v})
                    if not is_feasible(cand, sp.scenario):
                        continue
                    s = ev(cand)
                    if ev.out_of_budget:
                        return stop("budget")
                    if s is not None and (best is None or s < best):
                        best, moved = s, True
                        cur = cand
            if not moved:
                return stop("converged")
        return stop("done")
