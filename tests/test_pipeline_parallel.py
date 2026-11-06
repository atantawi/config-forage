"""Pipeline parallelism (pp): priced only on a stack that says so, never with dp > 1, and counted in
the instance width (tp·pp·dp) but not in the EP width (tp·dp)."""

from dataclasses import replace
from pathlib import Path

import pytest

from config_forage.blis import run_blis
from config_forage.blis.run_blis import ConfigError
from config_forage.blis.stacks import STACKS
from config_forage.feasibility import enumerate_feasible, feasible_topologies, is_feasible
from config_forage.space import load_space

V14, V15 = STACKS["v0.14.0"], STACKS["v0.15.0"]
PP_SPACE = Path(__file__).parent / "data" / "tiny-pp-space.yaml"
TINY = Path(__file__).parent / "data" / "tiny-space.yaml"


def test_only_v015_prices_pp():
    assert V15.pipeline_parallel and not V14.pipeline_parallel


def test_classify_pp_on_v015():
    p, _, _, _ = run_blis.classify_config({"tp": 8, "pp": 2, "enable_expert_parallel": True}, "0.32.0", V15)
    assert (p["tp"], p["pp"], p["dp"]) == (8, 2, 1)
    assert run_blis.footprint_gpus(p) == 16


def test_classify_pp_refused_on_v014():
    with pytest.raises(ConfigError, match="does not price pipeline parallelism"):
        run_blis.classify_config({"tp": 4, "pp": 2}, "0.29.0", V14)
    assert run_blis.classify_config({"tp": 4, "pp": 1}, "0.29.0", V14)[0]["pp"] == 1


def test_classify_pp_with_dp_refused():
    with pytest.raises(ConfigError, match="pp 2 with dp 2"):
        run_blis.classify_config({"tp": 4, "pp": 2, "dp": 2, "enable_expert_parallel": True}, "0.32.0", V15)


@pytest.mark.parametrize("pp", [0, -1, 2.0, "2", True])
def test_classify_pp_must_be_a_positive_int(pp):
    with pytest.raises(ConfigError, match="'pp': must be a positive integer"):
        run_blis.classify_config({"tp": 8, "pp": pp}, "0.32.0", V15)


def test_eplb_ep_width_excludes_pp(monkeypatch):
    """EP shards over tp·dp (blis-schemas ExpertParallelWidth), so pp must not widen the EPLB check."""
    monkeypatch.setattr(run_blis, "model_experts", lambda catalog, model: 256)
    eplb = {"eplb": {"enable_eplb": True, "num_redundant_experts": 8}}
    # EP width 8 divides 264; with pp counted it would be 16, which does not
    run_blis.check_eplb_layout("cat", "m", {"tp": 8, "pp": 2, "dp": 1}, eplb)
    with pytest.raises(ConfigError, match="width 16 does not divide"):
        run_blis.check_eplb_layout("cat", "m", {"tp": 8, "pp": 1, "dp": 2}, eplb)


def test_is_feasible_pp():
    sc = load_space(PP_SPACE).scenario
    assert is_feasible({"tp": 8, "pp": 2, "enable_expert_parallel": True}, sc)     # 16 GPUs, 1 instance
    assert is_feasible({"tp": 4, "pp": 2, "enable_expert_parallel": True}, sc)     # 8, 2 instances
    assert is_feasible({"tp": 2, "pp": 8}, sc)
    assert not is_feasible({"tp": 2, "pp": 2}, sc)                                 # 4 < min_footprint 8
    assert not is_feasible({"tp": 4, "pp": 2, "dp": 2, "enable_expert_parallel": True}, sc)  # pp x dp
    assert not is_feasible({"tp": 16, "pp": 2}, sc)                                # 32 > budget 16
    v14 = replace(sc, stack="v0.14.0", engine_version="0.29.0")
    assert not is_feasible({"tp": 8, "pp": 2}, v14) and is_feasible({"tp": 8}, v14)


def test_ep_over_pp_only_is_infeasible():
    """EP needs tp·dp > 1; pipeline stages alone shard no experts."""
    sc = replace(load_space(PP_SPACE).scenario, model_entry={"arch": "MoE"})
    assert not is_feasible({"tp": 1, "pp": 8, "enable_expert_parallel": True}, sc)
    assert is_feasible({"tp": 1, "pp": 8}, sc)


def test_space_config_counts_pp_in_width():
    sp = load_space(PP_SPACE)
    c = sp.config(sp.decision({"tp": 4, "pp": 2, "dp": 1, "enable_expert_parallel": True, "max_num_seqs": 64}))
    assert c["num_instances"] == 2
    c = sp.config(sp.decision({"tp": 2, "pp": 8, "dp": 1, "enable_expert_parallel": True, "max_num_seqs": 64}))
    assert c["num_instances"] == 1


def test_pp_space_topologies():
    sp = load_space(PP_SPACE)
    got = {(t["tp"], t["pp"], t["dp"], t["enable_expert_parallel"]) for t in feasible_topologies(sp)}
    want = set()
    for tp in (2, 4, 8, 16):
        for pp in (1, 2, 4, 8):
            for dp in (1, 2):
                for ep in (False, True):
                    w = tp * pp * dp
                    if 16 % w or w < 8 or (pp > 1 and dp > 1) or 8 % tp:  # heads_kv 8
                        continue
                    want.add((tp, pp, dp, ep))
    assert got == want
    assert (8, 2, 1, True) in got and (2, 8, 1, False) in got and (4, 2, 2, True) not in got
    assert len(list(enumerate_feasible(sp))) == 2 * len(got)


def test_spaces_without_pp_unchanged():
    """A space with no pp knob enumerates exactly as before pp joined the topology."""
    sp = load_space(TINY)
    assert all("pp" not in t for t in feasible_topologies(sp))
    assert all(is_feasible({**sp.knob_values(d), "pp": 1}, sp.scenario) for d in enumerate_feasible(sp))
