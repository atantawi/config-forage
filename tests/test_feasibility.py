from itertools import product
from pathlib import Path

import numpy as np

from config_forage.bsa.generator import CategoricalGenerator
from config_forage.feasibility import FeasibilityBiaser, enumerate_feasible, is_feasible
from config_forage.space import load_space

TINY = Path(__file__).parent / "data" / "tiny-space.yaml"


def test_is_feasible_rules():
    sp = load_space(TINY)
    moe = sp.scenario
    dense = type(moe)(**{**moe.__dict__, "model_entry": {"arch": "dense", "heads_kv": 8}})
    assert is_feasible({"tp": 4, "dp": 2, "enable_expert_parallel": True}, moe)
    assert not is_feasible({"tp": 1, "dp": 2}, dense)  # dense: no dp
    assert not is_feasible({"tp": 2, "enable_expert_parallel": True}, dense)  # dense: no EP
    assert not is_feasible({"tp": 1, "dp": 1, "enable_expert_parallel": True}, moe)
    assert is_feasible({"tp": 8, "dp": 2}, moe)  # 16 = two whole nodes (one multi-node instance)
    big = type(moe)(**{**moe.__dict__, "gpu_budget": 24})
    assert not is_feasible({"tp": 4, "dp": 3}, big)  # 12 straddles a node boundary
    assert is_feasible({"tp": 8, "dp": 3}, big) and not is_feasible({"tp": 8, "dp": 2}, big)  # 24 % 16
    low = type(moe)(**{**moe.__dict__, "gpu_budget": 12})
    assert not is_feasible({"tp": 4, "dp": 2}, low)  # 12 % 8
    assert is_feasible({"tp": 4}, low)
    kv = type(moe)(**{**moe.__dict__, "model_entry": {"arch": "MoE", "heads_kv": 4}})
    assert not is_feasible({"tp": 8}, kv) and is_feasible({"tp": 4}, kv)
    assert is_feasible({"tp": 8}, type(moe)(**{**moe.__dict__, "model_entry": {}}))  # no head check
    assert is_feasible({}, moe)  # absent topology knobs


def test_min_footprint_floor():
    sp = load_space(TINY)
    floor = type(sp.scenario)(**{**sp.scenario.__dict__,
                                 "model_entry": {"arch": "MoE", "heads_kv": 8, "min_footprint": 8}})
    assert is_feasible({"tp": 4, "dp": 2}, floor) and is_feasible({"tp": 8}, floor)
    assert not is_feasible({"tp": 4}, floor) and not is_feasible({"tp": 2, "dp": 2}, floor)
    assert not is_feasible({}, floor)  # absent topology = width 1
    fsp = type(sp)(floor, sp.knobs)
    feas = list(enumerate_feasible(fsp))
    assert feas and all(fsp.knob_values(d)["tp"] * fsp.knob_values(d)["dp"] in (8, 16) for d in feas)
    assert len(feas) < sum(1 for _ in enumerate_feasible(sp))
    g, rng = CategoricalGenerator(fsp.decision_space), np.random.default_rng(0)
    for _ in range(300):
        assert is_feasible(fsp.config(g.sample(rng, [FeasibilityBiaser(fsp)])), floor)


def test_feasibility_biaser_never_samples_infeasible():
    sp = load_space(TINY)
    g, rng = CategoricalGenerator(sp.decision_space), np.random.default_rng(0)
    seen = set()
    for _ in range(2000):
        d = g.sample(rng, [FeasibilityBiaser(sp)])
        assert is_feasible(sp.knob_values(d), sp.scenario)
        seen.add(d)
    assert len(seen) > 20


def test_enumerate_feasible_matches_filter():
    sp = load_space(TINY)
    got = set(enumerate_feasible(sp))
    want = set()
    for d in product(*(range(n) for n in sp.decision_space.sizes)):
        d = list(d)
        if d[4] != 1:
            d[5] = None
        want.add(tuple(d))
    want = {d for d in want if is_feasible(sp.knob_values(d), sp.scenario)}
    assert got == want and len(got) > 0


# --- P1: pruning on the topology prefix ------------------------------------------------------------

SPACES = Path(__file__).resolve().parent.parent / "spaces"
MID = SPACES / "glm53flash-h100-chatbot-16.yaml"
LARGE = SPACES / "glm53flash-h100-chatbot-16-large.yaml"


def _brute(sp):
    """The pre-pruning enumeration: every applicability-consistent leaf in DFS order, then filtered."""
    names = sp.names

    def rec(d):
        i = len(d)
        if i == len(sp.knobs):
            if is_feasible(sp.knob_values(d), sp.scenario):
                yield d
            return
        k = sp.knobs[i]
        if k.requires is not None:
            j = d[names.index(k.requires[0])]
            if j is None or sp.knobs[names.index(k.requires[0])].values[j] not in k.requires[1]:
                yield from rec(d + (None,))
                return
        for j in range(len(k.values)):
            yield from rec(d + (j,))

    return list(rec(()))


def _topology_last(sp):
    """The same space with the topology knobs moved after every other knob."""
    from config_forage.feasibility import TOPOLOGY
    from config_forage.space import KnobSpace
    return KnobSpace(sp.scenario, tuple([k for k in sp.knobs if k.name not in TOPOLOGY]
                                        + [k for k in sp.knobs if k.name in TOPOLOGY]))


def test_enumerate_feasible_order_unchanged():
    for sp in (load_space(TINY), load_space(MID), _topology_last(load_space(TINY))):
        assert list(enumerate_feasible(sp)) == _brute(sp)


def test_enumerate_feasible_first_on_s_large_is_fast():
    from helpers import Deadline
    sp = load_space(LARGE)
    with Deadline(0.5):
        d = next(enumerate_feasible(sp))
    assert is_feasible(sp.knob_values(d), sp.scenario)


def test_feasible_topologies():
    from config_forage.feasibility import feasible_topologies
    from helpers import Deadline
    for sp in (load_space(TINY), load_space(MID), _topology_last(load_space(TINY))):
        want = []
        for d in _brute(sp):
            t = {n: v for n, v in sp.knob_values(d).items() if n in ("tp", "dp", "enable_expert_parallel")}
            if t not in want:
                want.append(t)
        assert feasible_topologies(sp) == want
    with Deadline(0.5):
        ts = feasible_topologies(load_space(LARGE))
    assert ts and all(t["tp"] * t["dp"] in (8, 16) for t in ts)
