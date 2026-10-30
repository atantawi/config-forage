"""Exact topology feasibility (rules from blis-config-search topologies.py) and its biaser.

Multi-node instances are allowed as BLIS v0.14.0 places them: a width tp·dp above one node must fill
whole nodes. Upstream's topologies.py keeps one instance per node unless a model does not fit."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product

import numpy as np

from config_forage.blis import run_blis
from config_forage.bsa.biaser import ApplicabilityBiaser
from config_forage.bsa.space import Decision
from config_forage.space import KnobSpace, Scenario

GPUS_PER_NODE = run_blis.GPUS_PER_NODE  # one source: run_blis refuses a straddle with the same rule
TOPOLOGY = ("tp", "dp", "enable_expert_parallel")


def is_feasible(config: dict, scenario: Scenario) -> bool:
    """Absent tp/dp count as 1, absent EP as false. The weights-fit check is left to BLIS, except for an
    optional `model_entry.min_footprint`: the smallest tp·dp whose per-rank memory fits (measured).

    Invariant: the result depends only on the topology knobs (TOPOLOGY) and the scenario, never on
    another knob. enumerate_feasible, feasible_topologies, FeasibilityBiaser and the SMAC forbiddens
    rely on it; a rule on another knob must also change them."""
    tp, dp, ep = config.get("tp", 1), config.get("dp", 1), config.get("enable_expert_parallel", False)
    width = tp * dp
    if scenario.model_entry.get("arch") != "MoE" and (dp != 1 or ep):
        return False  # dense models take neither dp nor EP
    if ep and width <= 1:
        return False  # EP over width 1 shards nothing
    if scenario.gpu_budget % width:
        return False
    if width > GPUS_PER_NODE and width % GPUS_PER_NODE:
        return False  # a multi-node instance fills whole nodes (BLIS places it; run_blis refuses a straddle)
    if width < scenario.model_entry.get("min_footprint", 1):
        return False  # weights + fixed occupancy do not fit below this width
    heads_kv = scenario.model_entry.get("heads_kv")
    return not (heads_kv and heads_kv % tp)


class FeasibilityBiaser:
    """Zeros options that no feasible topology completion allows; `requires` as ApplicabilityBiaser."""

    def __init__(self, space: KnobSpace):
        self.space = space
        names = space.names
        self.applic = space.applicability_biaser() or ApplicabilityBiaser(space.decision_space, {})
        self.topo = [(names.index(n), n) for n in TOPOLOGY if n in names]

    def factor(self, axis: int, partial: Decision) -> np.ndarray | None:
        f = self.applic.factor(axis, partial)
        if f is None or axis not in {i for i, _ in self.topo}:
            return f
        knobs, sc = self.space.knobs, self.space.scenario
        fixed = {n: knobs[i].values[partial[i]] for i, n in self.topo if i < axis and partial[i] is not None}
        later = [(n, knobs[i].values) for i, n in self.topo if i > axis]
        for j, v in enumerate(knobs[axis].values):
            base = {**fixed, knobs[axis].name: v}
            if not any(is_feasible({**base, **dict(zip((n for n, _ in later), vs))}, sc)
                       for vs in product(*(vals for _, vals in later))):
                f[j] = 0.0
        return f


def feasible_topologies(space: KnobSpace) -> list[dict]:
    """The distinct feasible assignments of the space's topology knobs, as {name: value}, in the
    order enumerate_feasible first meets them (lexicographic in the space's knob order). Never walks
    the other knobs: at most |tp|·|dp|·|EP| checks (50 in the shipped spaces). Empty iff the space has no feasible decision."""
    topo = [k for k in space.knobs if k.name in TOPOLOGY]
    out = [dict(zip((k.name for k in topo), vs)) for vs in product(*(k.values for k in topo))]
    return [t for t in out if is_feasible(t, space.scenario)]


def enumerate_feasible(space: KnobSpace) -> Iterator[Decision]:
    """Every feasible decision, conditional knobs left unlabelled (None) where they do not apply.

    DFS in knob order. Since is_feasible depends only on the topology knobs, a branch is pruned as
    soon as the last topology knob (in knob order) is decided and its topology prefix is infeasible;
    no leaf check is needed. The output and its order equal filtering every leaf."""
    names = space.names
    topo = [i for i, n in enumerate(names) if n in TOPOLOGY]
    check_at = max(topo) + 1 if topo else 0  # depth at which every topology knob is decided

    def rec(d: tuple):
        i = len(d)
        if i == check_at and not is_feasible({names[m]: space.knobs[m].values[d[m]] for m in topo},
                                             space.scenario):
            return
        if i == len(space.knobs):
            yield d
            return
        k = space.knobs[i]
        if k.requires is not None:
            dep = space.knobs[names.index(k.requires[0])]
            j = d[names.index(k.requires[0])]
            if j is None or dep.values[j] not in k.requires[1]:
                yield from rec(d + (None,))
                return
        for j in range(len(k.values)):
            yield from rec(d + (j,))

    yield from rec(())
