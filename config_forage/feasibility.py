"""Exact topology feasibility (rules from blis-config-search topologies.py) and its biaser."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product

import numpy as np

from config_forage.bsa.biaser import ApplicabilityBiaser
from config_forage.bsa.space import Decision
from config_forage.space import KnobSpace, Scenario

GPUS_PER_NODE = 8
TOPOLOGY = ("tp", "dp", "enable_expert_parallel")


def is_feasible(config: dict, scenario: Scenario) -> bool:
    """Absent tp/dp count as 1, absent EP as false. The weights-fit check is left to BLIS."""
    tp, dp, ep = config.get("tp", 1), config.get("dp", 1), config.get("enable_expert_parallel", False)
    width = tp * dp
    if scenario.model_entry.get("arch") != "MoE" and (dp != 1 or ep):
        return False  # dense models take neither dp nor EP
    if ep and width <= 1:
        return False  # EP over width 1 shards nothing
    if width > GPUS_PER_NODE or scenario.gpu_budget % width:
        return False
    heads_kv = scenario.model_entry.get("heads_kv")
    return not (heads_kv and heads_kv % tp)


class FeasibilityBiaser:
    """Zeros options that no feasible topology completion allows; `requires` as ApplicabilityBiaser."""

    def __init__(self, space: KnobSpace):
        self.space = space
        names = space.names
        req = {m: (names.index(k.requires[0]),
                   tuple(space.knobs[names.index(k.requires[0])].values.index(v) for v in k.requires[1]))
               for m, k in enumerate(space.knobs) if k.requires}
        self.applic = ApplicabilityBiaser(space.decision_space, req)
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


def enumerate_feasible(space: KnobSpace) -> Iterator[Decision]:
    """Every feasible decision, conditional knobs left unlabelled (None) where they do not apply."""
    names = space.names

    def rec(d: tuple):
        i = len(d)
        if i == len(space.knobs):
            if is_feasible({k.name: k.values[j] for k, j in zip(space.knobs, d) if j is not None}, space.scenario):
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
