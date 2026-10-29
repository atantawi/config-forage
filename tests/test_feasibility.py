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
    assert not is_feasible({"tp": 8, "dp": 2}, moe)  # 16 > one node
    low = type(moe)(**{**moe.__dict__, "gpu_budget": 12})
    assert not is_feasible({"tp": 4, "dp": 2}, low)  # 12 % 8
    assert is_feasible({"tp": 4}, low)
    kv = type(moe)(**{**moe.__dict__, "model_entry": {"arch": "MoE", "heads_kv": 4}})
    assert not is_feasible({"tp": 8}, kv) and is_feasible({"tp": 4}, kv)
    assert is_feasible({"tp": 8}, type(moe)(**{**moe.__dict__, "model_entry": {}}))  # no head check
    assert is_feasible({}, moe)  # absent topology knobs


def test_feasibility_biaser_never_samples_infeasible():
    sp = load_space(TINY)
    g, rng = CategoricalGenerator(sp.decision_space), np.random.default_rng(0)
    seen = set()
    for _ in range(2000):
        d = g.sample(rng, [FeasibilityBiaser(sp)])
        assert is_feasible(sp.config(d), sp.scenario)
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
    want = {d for d in want if is_feasible(sp.config(d), sp.scenario)}
    assert got == want and len(got) > 0
