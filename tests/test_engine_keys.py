"""Engine keys BLIS v0.14.0 prices: classify_config places them, the scenario writes them, the cache
keys them; malformed values are refused before any run."""

import os
import re
from dataclasses import replace

import pytest

from config_forage.blis import run_blis
from config_forage.blis.runner import CachedRunner, cache_key
from config_forage.blis.stacks import STACKS
from config_forage.space import Scenario

import subprocess as _subprocess

_REAL_RUN = _subprocess.run
EP = {"tp": 4, "dp": 2, "enable_expert_parallel": True}
EPLB = {"enable_eplb": True, "num_redundant_experts": 32}
# (engine_version, stack): each stack with its own engine release, for behaviour meant to match on both
ON = pytest.mark.parametrize("ev, stack", [("0.29.0", STACKS["v0.14.0"]), ("0.32.0", STACKS["v0.15.0"])],
                             ids=["v0.14.0", "v0.15.0"])


@pytest.mark.parametrize("key, value", [("all2all_backend", "deepep_high_throughput"),
                                        ("long_prefill_token_threshold", 1024),
                                        ("cudagraph_mode", "FULL"),
                                        ("max_num_batched_tokens", 2048), ("max_num_seqs", 16)])
@ON
def test_flat_engine_key_goes_to_engine(key, value, ev, stack):
    parallel, engine, cli, router = run_blis.classify_config({**EP, key: value}, ev, stack)
    assert engine[key] == value and key not in parallel and key not in cli and not router


@ON
def test_eplb_block_goes_to_engine_as_a_mapping(ev, stack):
    _, engine, _, _ = run_blis.classify_config({**EP, "eplb": EPLB}, ev, stack)
    assert engine["eplb"] == EPLB


@pytest.mark.parametrize("field", ["window_size", "step_interval"])
def test_inert_eplb_fields_refused_with_the_measured_reason(field):
    with pytest.raises(run_blis.ConfigError, match=f"{field}.*inert in BLIS v0.14.0"):
        run_blis.classify_config({**EP, "eplb": {**EPLB, field: 1000}})
    with pytest.raises(run_blis.ConfigError, match=f"{field}.*inert in BLIS v0.15.0"):
        run_blis.classify_config({**EP, "eplb": {**EPLB, field: 1000}}, "0.32.0", STACKS["v0.15.0"])


LEGAL = run_blis.ENGINE_RELEASES["0.29.0"]["all2all_backends"]


@pytest.mark.parametrize("bad", ["deepep_ht", "DeepEP_V2", "", 3, None, True])
@ON
def test_illegal_all2all_backend_refused_with_the_legal_list(bad, ev, stack):
    with pytest.raises(run_blis.ConfigError, match=rf"not accepted by engine {re.escape(ev)}; legal = .*deepep_v2"):
        run_blis.classify_config({**EP, "all2all_backend": bad}, ev, stack)


@pytest.mark.parametrize("ok", ["deepep_high_throughput", "deepep_low_latency", "deepep_v2", "nixl_ep"])
@ON
def test_legal_all2all_backend_accepted(ok, ev, stack):
    assert ok in LEGAL and ok in run_blis.ENGINE_RELEASES[ev]["all2all_backends"]
    assert run_blis.classify_config({**EP, "all2all_backend": ok}, ev, stack)[1]["all2all_backend"] == ok


# a stack whose engine release has no ENGINE_RELEASES entry (a real stack refuses any other version)
ODD = replace(STACKS["v0.14.0"], name="odd", engine_version="9.9.9")


def test_all2all_backend_on_an_unknown_engine_version_refused():
    with pytest.raises(run_blis.ConfigError, match=r"no legal backend list for engine_version '9\.9\.9'"):
        run_blis.classify_config({**EP, "all2all_backend": "deepep_v2"}, "9.9.9", ODD)
    run_blis.classify_config({**EP}, "9.9.9", ODD)       # nothing to check without the key


def test_run_one_checks_against_its_engine_version(tmp_path):
    rec = run_blis.run_one("blis", None, None, str(tmp_path), "m", "h100", "9.9.9", None, 1, 0, None,
                           {**EP, "all2all_backend": "deepep_v2"}, stack=ODD)
    assert rec["returncode"] is None and "no legal backend list" in rec["error"]


@pytest.mark.parametrize("cfg, match", [({"long_prefill_token_threshold": 32769}, "exceeds max_model_len 32768"),
                                        ({"long_prefill_token_threshold": 4097, "max_model_len": 4096},
                                         "exceeds max_model_len 4096"),
                                        ({"long_prefill_token_threshold": "1024"}, "non-negative integer"),
                                        ({"long_prefill_token_threshold": True}, "non-negative integer"),
                                        ({"long_prefill_token_threshold": False}, "non-negative integer"),
                                        ({"long_prefill_token_threshold": -1}, "non-negative integer"),
                                        ({"long_prefill_token_threshold": 1024.0}, "non-negative integer")])
@ON
def test_bad_long_prefill_token_threshold_refused(cfg, match, ev, stack):
    with pytest.raises(run_blis.ConfigError, match=match):
        run_blis.classify_config({**EP, **cfg}, ev, stack)


@pytest.mark.parametrize("cfg", [{"long_prefill_token_threshold": 32768},
                                 {"long_prefill_token_threshold": 4096, "max_model_len": 4096}])
@ON
def test_long_prefill_token_threshold_at_max_model_len_accepted(cfg, ev, stack):
    _, engine, _, _ = run_blis.classify_config({**EP, **cfg}, ev, stack)
    assert engine["long_prefill_token_threshold"] == cfg["long_prefill_token_threshold"]


SC = Scenario("m", "h100", "chatbot", 8, "0.29.0")
BASE = {**EP, "num_instances": 1}


@pytest.mark.parametrize("noop, canonical", [
    ({"eplb": {"enable_eplb": False}}, {}),
    ({"eplb": {"enable_eplb": False, "num_redundant_experts": 32}}, {}),
    ({"all2all_backend": "allgather_reducescatter"}, {}),
    ({"all2all_backend": "pplx"}, {}),
    ({"all2all_backend": "naive"}, {}),
    ({"long_prefill_token_threshold": 0}, {}),
    ({"eplb": {"enable_eplb": True, "num_redundant_experts": 0}}, {"eplb": {"enable_eplb": True}}),
])
def test_no_op_values_share_the_canonical_run(tmp_path, noop, canonical):
    assert run_blis.canonical_config({**BASE, **noop}) == {**BASE, **canonical}
    calls = []

    def fn(config, c, seed):
        calls.append(config)
        return {"metrics": {}, "error": None, "returncode": 0, "wall_s": 0.0, "deployment": {}}
    r = CachedRunner(None, SC, tmp_path, run_fn=fn)
    a, b = r.run({**BASE, **noop}, 8), r.run({**BASE, **canonical}, 8)
    assert a["cache_key"] == b["cache_key"] == cache_key({**BASE, **canonical}, 8, SC.seed, SC)
    assert len(calls) == 1
    files = set()
    for cfg in ({**BASE, **noop}, {**BASE, **canonical}):
        parallel, engine, _, _ = run_blis.classify_config(cfg)
        files.add(run_blis.write_scenario(tmp_path, "m", "h100", "0.29.0", parallel, engine))
    assert len(files) == 1


def test_malformed_no_op_block_is_still_refused(tmp_path):
    bad = {**BASE, "eplb": {"enable_eplb": False, "bogus": 1}}
    assert run_blis.canonical_config(bad) == bad
    rec = run_blis.run_one("blis", None, None, str(tmp_path), "m", "h100", "0.29.0", None, 1, 0, None, bad)
    assert rec["returncode"] is None and "bogus" in rec["error"]


@ON
def test_priced_values_are_not_normalised(ev, stack):
    for cfg in ({"eplb": {"enable_eplb": True}}, {"eplb": EPLB}, {"all2all_backend": "deepep_v2"},
                {"long_prefill_token_threshold": 64}):
        assert run_blis.canonical_config({**BASE, **cfg}, ev, stack) == {**BASE, **cfg}


def test_eplb_key_order_does_not_change_the_cache_key():
    a = {**BASE, "eplb": {"enable_eplb": True, "num_redundant_experts": 32}}
    b = {**BASE, "eplb": {"num_redundant_experts": 32, "enable_eplb": True}}
    assert cache_key(a, 8, 0, SC) == cache_key(b, 8, 0, SC)


@pytest.mark.parametrize("bad, match", [("yes", "must be a mapping"), ({}, "enable_eplb"),
                                        ({"num_redundant_experts": 32}, "enable_eplb"),
                                        ({"enable_eplb": "true"}, "enable_eplb"),
                                        ({**EPLB, "bogus": 1}, "bogus"),
                                        ({**EPLB, "num_redundant_experts": -1}, "num_redundant_experts"),
                                        ({**EPLB, "num_redundant_experts": True}, "num_redundant_experts"),
                                        ({**EPLB, "num_redundant_experts": "32"}, "num_redundant_experts")])
@ON
def test_malformed_eplb_block_refused(bad, match, ev, stack):
    with pytest.raises(run_blis.ConfigError, match=match):
        run_blis.classify_config({**EP, "eplb": bad}, ev, stack)


@pytest.mark.parametrize("key", ["prefill_instances", "num_speculative_tokens", "routing_policy"])
@ON
def test_still_unsupported_keys_refused_with_reason(key, ev, stack):
    with pytest.raises(run_blis.ConfigError, match=run_blis.UNSUPPORTED[key][:20]):
        run_blis.classify_config({**EP, key: 1}, ev, stack)


@ON
def test_unknown_key_lists_the_nested_blocks(ev, stack):
    with pytest.raises(run_blis.ConfigError, match="unknown key.*'eplb'"):
        run_blis.classify_config({**EP, "no_such_knob": 1}, ev, stack)


@ON
def test_scenario_writes_eplb_as_a_sub_block(ev, stack):
    parallel, engine, _, _ = run_blis.classify_config(
        {**EP, "all2all_backend": "deepep_high_throughput", "eplb": EPLB}, ev, stack)
    y = run_blis.scenario_yaml("m", "h100", ev, parallel, engine, "n", coefficients=stack.coefficients)
    assert '      all2all_backend: "deepep_high_throughput"\n' in y
    assert "      eplb:\n        enable_eplb: true\n        num_redundant_experts: 32\n" in y


@ON
def test_new_keys_change_the_scenario_file_and_the_cache_key(tmp_path, ev, stack):
    sc = Scenario("m", "h100", "chatbot", 8, ev, stack=stack.name)
    base = {**EP, "num_instances": 1}
    variants = [base, {**base, "all2all_backend": "deepep_v2"}, {**base, "long_prefill_token_threshold": 64},
                {**base, "eplb": {"enable_eplb": True}},
                {**base, "eplb": EPLB}, {**base, "eplb": {**EPLB, "num_redundant_experts": 64}}]
    keys, files = set(), set()
    for v in variants:
        keys.add(cache_key(v, 8, 0, sc))
        parallel, engine, _, _ = run_blis.classify_config(v, ev, stack)
        files.add(run_blis.write_scenario(tmp_path, "m", "h100", ev, parallel, engine,
                                          coefficients=stack.coefficients))
    assert len(keys) == len(variants) and len(files) == len(variants)


# glm-5.3-flash does not run on engine 0.32.0 (docs/experiments/v015-folds.md); kimi-k2.5 on H200 does,
# with 8 redundant experts (32 leave no KV room) at gpu_memory_utilization 0.95
@pytest.mark.blis
@pytest.mark.skipif(not os.environ.get("BLIS_BIN"), reason="BLIS_BIN not set")
@pytest.mark.parametrize("stack, sc, extra_base, eplb", [
    ("v0.14.0", Scenario("glm-5.3-flash", "h100", "chatbot", 8, "0.29.0"), {}, EPLB),
    ("v0.15.0", Scenario("kimi-k2.5", "h200", "chatbot", 8, "0.32.0", stack="v0.15.0"),
     {"gpu_memory_utilization": 0.95}, {"enable_eplb": True, "num_redundant_experts": 8})])
def test_blis_prices_the_new_engine_keys(tmp_path, monkeypatch, stack, sc, extra_base, eplb):
    """On an EP tp4*dp2 instance, each accepted key, set off its default, moves the metrics."""
    from helpers import blis_env_for
    monkeypatch.setattr(run_blis.subprocess, "run", _REAL_RUN)     # undo the _no_blis stub (and for check_binary)
    env = blis_env_for(stack)
    env.scenarios_dir = tmp_path / "scenarios"
    runner = CachedRunner(env, sc, tmp_path / "cache")
    base = {**EP, "num_instances": 1, "quantization": "fp8", "cache_dtype": "fp8", **extra_base}
    m = {}
    for name, extra in [("default", {}), ("a2a", {"all2all_backend": "deepep_high_throughput"}),
                        ("lptt", {"long_prefill_token_threshold": 64}), ("eplb", {"eplb": eplb})]:
        r = runner.run({**base, **extra}, 32)
        assert r["error"] is None, r.get("stderr_tail")
        m[name] = r["metrics"]
    for name in ("a2a", "lptt", "eplb"):
        assert m[name] != m["default"], name


# --- eplb layouts: the EP width must divide the model's experts + redundant experts ---------------

def _catalog(tmp_path, model, graph):
    d = tmp_path / "catalog" / "models" / model
    d.mkdir(parents=True)
    (d / "graph.yaml").write_text(graph)
    return str(tmp_path / "catalog")


MOE_GRAPH = """layer_kinds:
  - nodes:
    - op: GroupedGEMM
      role: shared_experts
      experts: 1
    - op: GroupedGEMM
      role: experts
      experts: 60
      top_k: 4
"""


@pytest.fixture(autouse=True)
def _no_blis(monkeypatch):
    """run_one reaching BLIS records rc=7: these tests only ask whether it got that far."""
    import subprocess
    monkeypatch.setattr(run_blis.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 7, "", "stub"))


def _run(tmp_path, catalog, model, config, ev="0.29.0", stack=STACKS["v0.14.0"]):
    return run_blis.run_one("blis", catalog, "registry", str(tmp_path / "sc"), model, "h100", ev,
                            None, 1, 0, None, config, stack=stack)


@ON
def test_enabled_eplb_without_expert_parallel_refused(ev, stack):
    with pytest.raises(run_blis.ConfigError, match="eplb.*needs enable_expert_parallel"):
        run_blis.classify_config({"tp": 8, "eplb": {"enable_eplb": True}}, ev, stack)
    with pytest.raises(run_blis.ConfigError, match="eplb.*needs enable_expert_parallel"):
        run_blis.classify_config({"tp": 8, "enable_expert_parallel": False, "eplb": EPLB}, ev, stack)


def test_model_experts_reads_the_largest_routed_count(tmp_path):
    assert run_blis.model_experts(_catalog(tmp_path, "moe", MOE_GRAPH), "moe") == 60
    assert run_blis.model_experts(_catalog(tmp_path / "d", "dense", "layer_kinds: []\n"), "dense") == 0


@ON
def test_indivisible_eplb_layout_refused_with_the_numbers(tmp_path, ev, stack):
    cat = _catalog(tmp_path, "moe", MOE_GRAPH)
    rec = _run(tmp_path, cat, "moe", {**EP, "eplb": {"enable_eplb": True, "num_redundant_experts": 2}}, ev, stack)
    assert rec["returncode"] is None and rec["cmd"] is None
    assert "60 experts + 2 redundant = 62" in rec["error"] and "width 8" in rec["error"]
    rec = _run(tmp_path, cat, "moe", {**EP, "eplb": {"enable_eplb": True}}, ev, stack)   # omitted = 0 redundant
    assert "60 experts + 0 redundant = 60" in rec["error"]


@ON
def test_divisible_eplb_layout_reaches_blis(tmp_path, ev, stack):
    cat = _catalog(tmp_path, "moe", MOE_GRAPH)
    rec = _run(tmp_path, cat, "moe", {**EP, "eplb": {"enable_eplb": True, "num_redundant_experts": 4}}, ev, stack)
    assert rec["cmd"] is not None and "experts" not in (rec["error"] or "")   # 64 % 8 == 0


@pytest.mark.parametrize("why", ["dense", "no catalog", "no model entry"])
@ON
def test_eplb_that_cannot_be_validated_is_refused(tmp_path, why, ev, stack):
    cat = {"dense": _catalog(tmp_path, "dense", "layer_kinds: []\n"), "no catalog": None,
           "no model entry": str(tmp_path)}[why]
    rec = _run(tmp_path, cat, "dense", {**EP, "eplb": EPLB}, ev, stack)
    assert rec["returncode"] is None and "eplb" in rec["error"] and "cannot validate" in rec["error"]


def test_eplb_off_needs_no_catalog(tmp_path):
    rec = _run(tmp_path, None, "dense", {**EP, "eplb": {"enable_eplb": False}})
    assert rec["cmd"] is not None


@pytest.mark.skipif(not os.environ.get("BLIS_CATALOG"), reason="BLIS_CATALOG not set")
def test_real_catalog_glm53_flash_has_288_experts(tmp_path):
    cat = os.environ["BLIS_CATALOG"]
    assert run_blis.model_experts(cat, "glm-5.3-flash") == 288
    ok = _run(tmp_path, cat, "glm-5.3-flash", {**EP, "eplb": EPLB})        # 320 % 8 == 0
    assert ok["cmd"] is not None
    bad = _run(tmp_path, cat, "glm-5.3-flash", {**EP, "eplb": {"enable_eplb": True, "num_redundant_experts": 30}})
    assert "288 experts + 30 redundant = 318" in bad["error"]


# --- cache contract: configs using the keys accepted here never hit refusal rows cached before -----

KEY_AT_MAIN = "0cce6fcd158a6d91db7d986f797942a7cd289f8b744f94cbf55da7d3cd4098bf"  # cache_key at 3ef1737


def test_config_without_the_new_keys_keeps_its_key_from_main():
    assert cache_key({"tp": 8, "num_instances": 1}, 8, 0, SC) == KEY_AT_MAIN


@pytest.mark.parametrize("extra", [{"all2all_backend": "deepep_v2"}, {"long_prefill_token_threshold": 64},
                                   {"eplb": {"enable_eplb": True}}])
def test_config_with_a_new_key_gets_a_new_key(extra):
    import hashlib
    import json

    from config_forage.blis.runner import PINS, requests_for
    cfg = {**BASE, **extra}
    old = {"config": cfg, "concurrency": 8, "seed": 0, "workload": SC.workload,
           "num_requests": requests_for(8, SC.num_requests_base), "model": SC.model,
           "hardware": SC.hardware, "engine_version": SC.engine_version, "pins": PINS}
    old_key = hashlib.sha256(json.dumps(old, sort_keys=True, default=str).encode()).hexdigest()
    assert cache_key(cfg, 8, 0, SC) != old_key
