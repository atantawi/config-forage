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


def test_eplb_takes_every_field_blis_names():
    block = {**EPLB, "window_size": 1000, "step_interval": 3000}
    assert run_blis.classify_config({**EP, "eplb": block})[1]["eplb"] == block


@pytest.mark.parametrize("bad, match", [("yes", "must be a mapping"), ({}, "enable_eplb"),
                                        ({"num_redundant_experts": 32}, "enable_eplb"),
                                        ({"enable_eplb": "true"}, "enable_eplb"),
                                        ({**EPLB, "bogus": 1}, "bogus"),
                                        ({**EPLB, "num_redundant_experts": -1}, "num_redundant_experts"),
                                        ({**EPLB, "window_size": 1.5}, "window_size")])
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
