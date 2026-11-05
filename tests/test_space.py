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


def test_space_hash_is_content_not_bytes(tmp_path):
    from config_forage.space import space_hash
    sp = load_space(TINY)
    h = space_hash(sp)
    assert len(h) == 64 and h == space_hash(load_space(TINY))
    noisy = tmp_path / "noisy.yaml"
    noisy.write_text("# a comment\n\n" + TINY.read_text().replace(", ", ",   "))
    assert space_hash(load_space(noisy)) == h  # comments and whitespace do not matter
    other = KnobSpace(sp.scenario, tuple(replace(k, values=k.values[:2]) if k.name == "max_num_seqs" else k
                                         for k in sp.knobs))
    assert space_hash(other) != h
    assert space_hash(KnobSpace(replace(sp.scenario, gpu_budget=8), sp.knobs)) != h
    assert space_hash(KnobSpace(sp.scenario, tuple(replace(k, default=k.values[-1]) if k.name == "tp" else k
                                                   for k in sp.knobs))) != h


def _ladder_yaml(tmp_path, **extra):
    import yaml
    raw = yaml.safe_load(TINY.read_text())
    raw["scenario"].update(extra)
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_ladder_fields_load(tmp_path):
    sc = load_space(_ladder_yaml(tmp_path, scoring="ladder", concurrency=[64, 256, 1024, 2048],
                                 early_stop=False)).scenario
    assert sc.scoring == "ladder" and sc.concurrency == (64, 256, 1024, 2048) and sc.early_stop is False
    hash(sc.concurrency)  # a tuple, not a list
    assert Slo(e2e_p99_ms=5000).passes({"itl_p99_ms": 1, "e2e_p99_ms": 4000})
    assert not Slo(e2e_p99_ms=5000).passes({"itl_p99_ms": 1, "e2e_p99_ms": 6000})
    assert not Slo(e2e_p99_ms=5000).passes({"itl_p99_ms": 1})


def test_ladder_needs_concurrency(tmp_path):
    with pytest.raises(ValueError, match="concurrency"):
        load_space(_ladder_yaml(tmp_path, scoring="ladder"))


@pytest.mark.parametrize("conc", [[256, 64], [64, 64], [0, 64], [-1, 64], [1.5, 64], [], [True, 64]])
def test_ladder_concurrency_must_be_ascending(tmp_path, conc):
    with pytest.raises(ValueError, match="concurrency"):
        load_space(_ladder_yaml(tmp_path, scoring="ladder", concurrency=conc))


def test_concurrency_with_probe_refused(tmp_path):
    with pytest.raises(ValueError, match="ambiguous"):
        load_space(_ladder_yaml(tmp_path, concurrency=[64, 256]))


def test_unknown_scoring_refused(tmp_path):
    with pytest.raises(ValueError, match="scoring"):
        load_space(_ladder_yaml(tmp_path, scoring="sweep"))


def test_hash_unchanged_by_defaults():
    from config_forage.space import space_hash
    from tests.test_stacks import ROOT, SPACE_HASHES
    for name, h in SPACE_HASHES.items():
        assert space_hash(load_space(ROOT / name)) == h


def test_ladder_fields_change_hash(tmp_path):
    from config_forage.space import space_hash
    base = space_hash(load_space(TINY))
    assert space_hash(load_space(_ladder_yaml(tmp_path, scoring="ladder", concurrency=[64, 256]))) != base
    assert space_hash(load_space(_ladder_yaml(tmp_path, scoring="ladder", concurrency=[64, 256], early_stop=False))) != base
    assert space_hash(load_space(_ladder_yaml(tmp_path, slo={"itl_p99_ms": 20, "e2e_p99_ms": 9000}))) != base


def test_throughput_per_gpu_slo_refused(tmp_path):
    with pytest.raises(ValueError, match="slo.throughput_per_gpu is not supported"):
        load_space(_ladder_yaml(tmp_path, slo={"itl_p99_ms": 20, "throughput_per_gpu": 5}))


@pytest.mark.parametrize("v", ["false", 0, 1, None])
def test_early_stop_must_be_bool(tmp_path, v):
    with pytest.raises(ValueError, match="early_stop must be a bool"):
        load_space(_ladder_yaml(tmp_path, scoring="ladder", concurrency=[64, 256], early_stop=v))


def test_early_stop_with_probe_refused(tmp_path):
    # the default (true) is indistinguishable from absent, so only a non-default value is caught
    with pytest.raises(ValueError, match="early_stop with scoring 'probe'"):
        load_space(_ladder_yaml(tmp_path, early_stop=False))
