"""Engine keys BLIS v0.14.0 prices: classify_config places them, the scenario writes them, the cache
keys them; malformed values are refused before any run."""

import os

import pytest

from config_forage.blis import run_blis
from config_forage.blis.runner import BlisEnv, CachedRunner, cache_key
from config_forage.space import Scenario

EP = {"tp": 4, "dp": 2, "enable_expert_parallel": True}
EPLB = {"enable_eplb": True, "num_redundant_experts": 32}


@pytest.mark.parametrize("key, value", [("all2all_backend", "deepep_high_throughput"),
                                        ("long_prefill_token_threshold", 1024),
                                        ("cudagraph_mode", "FULL"),
                                        ("max_num_batched_tokens", 2048), ("max_num_seqs", 16)])
def test_flat_engine_key_goes_to_engine(key, value):
    parallel, engine, cli, router = run_blis.classify_config({**EP, key: value})
    assert engine[key] == value and key not in parallel and key not in cli and not router


def test_eplb_block_goes_to_engine_as_a_mapping():
    _, engine, _, _ = run_blis.classify_config({**EP, "eplb": EPLB})
    assert engine["eplb"] == EPLB


@pytest.mark.parametrize("field", ["window_size", "step_interval"])
def test_inert_eplb_fields_refused_with_the_measured_reason(field):
    with pytest.raises(run_blis.ConfigError, match=f"{field}.*inert in BLIS v0.14.0"):
        run_blis.classify_config({**EP, "eplb": {**EPLB, field: 1000}})


LEGAL = run_blis.ENGINE_RELEASES["0.29.0"]["all2all_backends"]


@pytest.mark.parametrize("bad", ["deepep_ht", "DeepEP_V2", "", 3, None, True])
def test_illegal_all2all_backend_refused_with_the_legal_list(bad):
    with pytest.raises(run_blis.ConfigError, match=r"not accepted by engine 0\.29\.0; legal = .*deepep_v2"):
        run_blis.classify_config({**EP, "all2all_backend": bad})


@pytest.mark.parametrize("ok", ["deepep_high_throughput", "deepep_low_latency", "deepep_v2", "nixl_ep"])
def test_legal_all2all_backend_accepted(ok):
    assert ok in LEGAL
    assert run_blis.classify_config({**EP, "all2all_backend": ok})[1]["all2all_backend"] == ok


def test_all2all_backend_on_an_unknown_engine_version_refused():
    with pytest.raises(run_blis.ConfigError, match=r"no legal backend list for engine_version '9\.9\.9'"):
        run_blis.classify_config({**EP, "all2all_backend": "deepep_v2"}, "9.9.9")
    run_blis.classify_config({**EP}, "9.9.9")       # nothing to check without the key


def test_run_one_checks_against_its_engine_version(tmp_path):
    rec = run_blis.run_one("blis", None, None, str(tmp_path), "m", "h100", "9.9.9", None, 1, 0, None,
                           {**EP, "all2all_backend": "deepep_v2"})
    assert rec["returncode"] is None and "no legal backend list" in rec["error"]


@pytest.mark.parametrize("cfg, match", [({"long_prefill_token_threshold": 32769}, "exceeds max_model_len 32768"),
                                        ({"long_prefill_token_threshold": 4097, "max_model_len": 4096},
                                         "exceeds max_model_len 4096"),
                                        ({"long_prefill_token_threshold": "1024"}, "non-negative integer"),
                                        ({"long_prefill_token_threshold": True}, "non-negative integer"),
                                        ({"long_prefill_token_threshold": False}, "non-negative integer"),
                                        ({"long_prefill_token_threshold": -1}, "non-negative integer"),
                                        ({"long_prefill_token_threshold": 1024.0}, "non-negative integer")])
def test_bad_long_prefill_token_threshold_refused(cfg, match):
    with pytest.raises(run_blis.ConfigError, match=match):
        run_blis.classify_config({**EP, **cfg})


@pytest.mark.parametrize("cfg", [{"long_prefill_token_threshold": 32768},
                                 {"long_prefill_token_threshold": 4096, "max_model_len": 4096}])
def test_long_prefill_token_threshold_at_max_model_len_accepted(cfg):
    _, engine, _, _ = run_blis.classify_config({**EP, **cfg})
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


def test_priced_values_are_not_normalised():
    for cfg in ({"eplb": {"enable_eplb": True}}, {"eplb": EPLB}, {"all2all_backend": "deepep_v2"},
                {"long_prefill_token_threshold": 64}):
        assert run_blis.canonical_config({**BASE, **cfg}) == {**BASE, **cfg}


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
def test_malformed_eplb_block_refused(bad, match):
    with pytest.raises(run_blis.ConfigError, match=match):
        run_blis.classify_config({**EP, "eplb": bad})


@pytest.mark.parametrize("key", ["prefill_instances", "num_speculative_tokens", "routing_policy"])
def test_still_unsupported_keys_refused_with_reason(key):
    with pytest.raises(run_blis.ConfigError, match=run_blis.UNSUPPORTED[key][:20]):
        run_blis.classify_config({**EP, key: 1})


def test_unknown_key_lists_the_nested_blocks():
    with pytest.raises(run_blis.ConfigError, match="unknown key.*'eplb'"):
        run_blis.classify_config({**EP, "no_such_knob": 1})


def test_scenario_writes_eplb_as_a_sub_block():
    parallel, engine, _, _ = run_blis.classify_config(
        {**EP, "all2all_backend": "deepep_high_throughput", "eplb": EPLB})
    y = run_blis.scenario_yaml("m", "h100", "0.29.0", parallel, engine, "n")
    assert '      all2all_backend: "deepep_high_throughput"\n' in y
    assert "      eplb:\n        enable_eplb: true\n        num_redundant_experts: 32\n" in y


def test_new_keys_change_the_scenario_file_and_the_cache_key(tmp_path):
    sc = Scenario("m", "h100", "chatbot", 8, "0.29.0")
    base = {**EP, "num_instances": 1}
    variants = [base, {**base, "all2all_backend": "deepep_v2"}, {**base, "long_prefill_token_threshold": 64},
                {**base, "eplb": {"enable_eplb": True}},
                {**base, "eplb": EPLB}, {**base, "eplb": {**EPLB, "num_redundant_experts": 64}}]
    keys, files = set(), set()
    for v in variants:
        keys.add(cache_key(v, 8, 0, sc))
        parallel, engine, _, _ = run_blis.classify_config(v)
        files.add(run_blis.write_scenario(tmp_path, "m", "h100", "0.29.0", parallel, engine))
    assert len(keys) == len(variants) and len(files) == len(variants)


@pytest.mark.blis
@pytest.mark.skipif(not os.environ.get("BLIS_BIN"), reason="BLIS_BIN not set")
def test_blis_prices_the_new_engine_keys(tmp_path):
    """On an EP tp4*dp2 instance, each accepted key, set off its default, moves the metrics."""
    env = BlisEnv.from_env()
    env.scenarios_dir = tmp_path / "scenarios"
    sc = Scenario("glm-5.3-flash", "h100", "chatbot", 8, "0.29.0")
    runner = CachedRunner(env, sc, tmp_path / "cache")
    base = {**EP, "num_instances": 1, "quantization": "fp8", "cache_dtype": "fp8"}
    m = {}
    for name, extra in [("default", {}), ("a2a", {"all2all_backend": "deepep_high_throughput"}),
                        ("lptt", {"long_prefill_token_threshold": 64}), ("eplb", {"eplb": EPLB})]:
        r = runner.run({**base, **extra}, 32)
        assert r["error"] is None, r.get("stderr_tail")
        m[name] = r["metrics"]
    for name in ("a2a", "lptt", "eplb"):
        assert m[name] != m["default"], name
