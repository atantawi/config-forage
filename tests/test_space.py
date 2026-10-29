from dataclasses import replace
from pathlib import Path

import pytest

from config_forage.space import Knob, KnobSpace, Slo, load_space

TINY = Path(__file__).parent / "data" / "tiny-space.yaml"


def test_load_space_roundtrip():
    sp = load_space(TINY)
    c = {"tp": 4, "dp": 2, "enable_expert_parallel": True, "max_num_seqs": 256,
         "speculative": True, "num_speculative_tokens": 3, "num_instances": 2}
    assert sp.config(sp.decision(c)) == c
    assert sp.decision_space.axes[0].options == ("1", "2", "4", "8")
    assert sp.n_combinations() == 4 * 4 * 2 * 3 * 2 * 2
    assert sp.scenario.slo == Slo(itl_p99_ms=20) and sp.scenario.num_requests_base == 80


def test_config_derives_num_instances():
    sp = load_space(TINY)
    assert sp.config((2, 1, 0, 0, 0, None))["num_instances"] == 2  # tp=4, dp=2, budget 16


def test_conditional_knob_unlabelled():
    sp = load_space(TINY)
    d = sp.decision({"tp": 1, "dp": 1, "enable_expert_parallel": False, "max_num_seqs": 64, "speculative": False})
    assert d[-1] is None and "num_speculative_tokens" not in sp.knob_values(d)


def test_default_config():
    sp = load_space(TINY)
    c = sp.default_config()
    assert c == {"tp": 2, "dp": 1, "enable_expert_parallel": False, "max_num_seqs": 128,
                 "speculative": False, "num_instances": 8}
    bad = KnobSpace(sp.scenario, (Knob("tp", (8,)), Knob("dp", (4,))))
    with pytest.raises(ValueError, match="infeasible"):
        bad.default_config()
    assert Knob("x", (3, 4)).default == 3
    with pytest.raises(ValueError):
        Knob("x", (3, 4), default=5)


def test_bad_requires_rejected():
    sp = load_space(TINY)
    with pytest.raises(ValueError):
        KnobSpace(sp.scenario, (Knob("a", (1,), requires=("b", (1,))), Knob("b", (1,))))


def test_slo_passes():
    s = Slo(itl_p99_ms=20, ttft_p99_ms=500)
    assert s.passes({"itl_p99_ms": 20, "ttft_p99_ms": 400})
    assert not s.passes({"itl_p99_ms": 21, "ttft_p99_ms": 400})
    assert not s.passes({"itl_p99_ms": 10})  # missing required metric
    assert Slo().passes({"itl_p99_ms": 5}) and not Slo().passes({})


def test_config_rejects_zero_instances():
    sc = load_space(TINY).scenario  # budget 16
    sp = KnobSpace(sc, (Knob("tp", (8,)), Knob("dp", (4,))))
    with pytest.raises(ValueError, match="num_instances"):
        sp.config((0, 0))
    assert sp.knob_values((0, 0)) == {"tp": 8, "dp": 4}


def test_topology_knobs_cannot_be_conditional():
    sc = load_space(TINY).scenario
    for name in ("tp", "dp", "enable_expert_parallel"):
        with pytest.raises(ValueError, match="topology"):
            KnobSpace(sc, (Knob("x", (0, 1)), Knob(name, (1, 2) if name != "enable_expert_parallel" else (False, True),
                                                   requires=("x", (1,)))))


def test_option_labels_must_be_unique():
    sc = load_space(TINY).scenario
    with pytest.raises(ValueError, match="label"):
        KnobSpace(sc, (Knob("x", (1, "1")),))
    with pytest.raises(ValueError, match="label"):
        KnobSpace(sc, (Knob("x", (1, 1)),))
