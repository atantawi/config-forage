import pytest

from config_forage import router
from config_forage.blis import run_blis
from config_forage.blis.runner import CachedRunner, cache_key
from config_forage.space import Scenario

SC = Scenario("m", "h100", "chatbot", 8, "0.29.0")


def weights(epp):
    (prof,) = epp["schedulingProfiles"]
    return {p["pluginRef"]: p.get("weight") for p in prof["plugins"]}


def test_family_matches_blis_example():
    epp = router.endpoint_picker_config("prefix-affinity")
    assert [p["type"] for p in epp["plugins"]] == ["single-profile-handler", "approx-prefix-cache-producer",
                                                   "prefix-cache-scorer", "queue-scorer",
                                                   "kv-cache-utilization-scorer", "max-score-picker"]
    assert weights(epp) == {"prefix-cache-scorer": 3, "queue-scorer": 1, "kv-cache-utilization-scorer": 1,
                            "max-score-picker": None}


def test_weight_override_and_zero():
    epp = router.endpoint_picker_config("load-only", {"queue-scorer": 0})
    assert weights(epp)["queue-scorer"] == 0 and weights(epp)["kv-cache-utilization-scorer"] == 1


def test_unknown_family_or_scorer_refused():
    with pytest.raises(ValueError, match="not one of"):
        router.endpoint_picker_config("round-robin")
    with pytest.raises(ValueError, match="no scorer"):
        router.endpoint_picker_config("load-only", {"prefix-cache-scorer": 1})


def test_to_blis():
    assert router.to_blis({"tp": 8}) == {"tp": 8}
    out = router.to_blis({"tp": 8, "endpoint_picker_config": "inflight-load", "active_request_scorer_weight": 4})
    assert set(out) == {"tp", "epp_config"} and weights(out["epp_config"])["active-request-scorer"] == 4
    with pytest.raises(ValueError, match="need endpoint_picker_config"):
        router.to_blis({"queue_scorer_weight": 2})


def test_cache_key_covers_epp_and_runner_passes_it(tmp_path):
    seen = []

    def fn(config, c, seed):
        seen.append(config)
        return {"metrics": {}, "error": None, "returncode": 0, "deployment": {}}
    r = CachedRunner(None, SC, tmp_path, run_fn=fn)
    a = {"tp": 8, "endpoint_picker_config": "load-only"}
    b = {**a, "queue_scorer_weight": 4}
    r.run(a, 8), r.run(b, 8), r.run(a, 8)
    assert len(seen) == 2 and all("epp_config" in s and "endpoint_picker_config" not in s for s in seen)
    assert cache_key(router.to_blis(a), 8, 0, SC) != cache_key(router.to_blis(b), 8, 0, SC)


def test_run_blis_writes_router_block(tmp_path):
    parallel, engine, cli, rt = run_blis.classify_config(
        {"tp": 8, "dp": 2, "data_parallel_load_balancing": "external", "num_instances": 1,
         "epp_config": router.endpoint_picker_config("load-only")})
    assert parallel["data_parallel_load_balancing"] == "external" and cli == {"num_instances": 1}
    epp = run_blis.write_epp(tmp_path, rt["epp_config"])
    name = run_blis.write_scenario(tmp_path, "m", "h100", "0.29.0", parallel, engine, 1, epp_file=epp)
    body = (tmp_path / name).read_text()
    assert body.endswith(f"router:\n  endpoint_picker_config: {epp}\n") and "fabric: ib-400g" in body
    assert (tmp_path / epp).exists()
    other = run_blis.write_scenario(tmp_path, "m", "h100", "0.29.0", parallel, engine, 1)
    assert other != name and "router:" not in (tmp_path / other).read_text()


@pytest.mark.parametrize("key", ["routing_policy", "routing_scorers", "admission_policy"])
def test_removed_blis_routing_flags_refused(key):
    with pytest.raises(run_blis.ConfigError, match="removed in BLIS v0.12.0"):
        run_blis.classify_config({key: "x"})
