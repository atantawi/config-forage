"""Stack v0.16.0 (inference-sim v0.16.0, Result schema 2, embedded catalog and registry).

Fixtures are real BLIS v0.16.0 stdout (3f4998e2, native arm64), captured through run_blis's scenario
file and command line, with `request_rows` emptied to keep them small (the parser does not read it):
  result-v016-kimi.json        kimi-k2.5/h200/chatbot, 1 x tp4*dp2 + EP, fp8, c=64, 256 requests, seed 42
  result-v016-kimi-2inst.json  the same with num_instances 2
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from config_forage.blis import result_doc, run_blis, runner
from config_forage.blis.runner import CachedRunner
from config_forage.blis.stacks import STACKS
from config_forage.space import Scenario

DATA = Path(__file__).parent / "data"
V16 = STACKS.get("v0.16.0")
EP = {"tp": 4, "dp": 2, "enable_expert_parallel": True}
CFG = {**EP, "num_instances": 1}


def _doc(name):
    return (DATA / name).read_text()


def _path(doc, dotted):
    for p in dotted.split("."):
        doc = doc[p]
    return doc


# flat key -> its schema 2 home (inference-sim v0.16.0 docs/guide/results.md, "Where each
# MetricsOutput field went"), spelled out here independently of the module
CITED_V2 = {
    "completed_requests": "requests.completed", "still_queued": "requests.queued_at_horizon",
    "still_running": "requests.in_flight_at_horizon", "timed_out_requests": "requests.timed_out",
    "length_capped_requests": "requests.length_capped",
    "total_input_tokens": "reports.engine.cluster.total_input_tokens",
    "total_output_tokens": "reports.engine.cluster.total_output_tokens",
    "vllm_estimated_duration_s": "reports.engine.cluster.simulated_duration_s",
    "responses_per_sec": "reports.engine.cluster.whole_run.responses_per_sec",
    "tokens_per_sec": "reports.engine.cluster.whole_run.tokens_per_sec",
    **{f"{f}_{s}_ms": f"reports.engine.cluster.whole_run.{f}_{s}_ms"
       for f in ("ttft", "e2e") for s in ("mean", "p90", "p95", "p99")},
    **{f"itl_{s}_ms": f"reports.engine.cluster.whole_run.step_interval_{s}_ms" for s in ("mean", "p90", "p95", "p99")},
    "scheduling_delay_p99_ms": "reports.engine.cluster.scheduling_delay_p99_ms",
    "kv_allocation_failures": "reports.engine.cluster.kv_allocation_failures",
    "preemption_count": "reports.engine.cluster.preemptions",
}


# --- the stack entry -------------------------------------------------------------------------------

def test_v016_stack_pins():
    assert V16 is not None
    assert V16.pins == ("inference-sim v0.16.0 3f4998e28f922611a2d21d5c7ffe18a5272fe114", "substrate v0.13.0",
                        "blis-vllm v0.32.0-blis.2", "router v1.0.0-blis.3", "data embedded",
                        "workload as token-shape flags")
    assert V16.sim_sha == "3f4998e28f922611a2d21d5c7ffe18a5272fe114"
    assert V16.engine_version == "0.32.0" and V16.pipeline_parallel
    assert V16.result_format == "result-doc-v2" and V16.embedded_data
    # the substrate checkout config-forage reads workload presets and model graphs from
    assert V16.catalog_tag == V16.registry_tag == "v0.13.0"
    assert V16.coefficients == STACKS["v0.15.0"].coefficients


def test_older_stacks_pass_their_data():
    assert not STACKS["v0.14.0"].embedded_data and not STACKS["v0.15.0"].embedded_data


# --- schema 2 parsing ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["result-v016-kimi.json", "result-v016-kimi-2inst.json"])
def test_v2_cluster_matches_document(name):
    doc = json.loads(_doc(name))
    m, _ = result_doc.parse_result_document(_doc(name), version=2)
    want = {"instance_id": "cluster", **{k: _path(doc, p) for k, p in CITED_V2.items()},
            "injected_requests": doc["requests"]["generated"], "dropped_unservable": 0}
    assert m == want
    assert m["injected_requests"] == m["completed_requests"] == 256


def test_v2_keys_are_the_v015_keys():
    v15, v15_inst = result_doc.parse_result_document(_doc("result-v015-kimi-2inst.json"))
    v16, v16_inst = result_doc.parse_result_document(_doc("result-v016-kimi-2inst.json"), version=2)
    assert set(v15) == set(v16)
    assert [set(i) for i in v15_inst] == [set(i) for i in v16_inst]


def test_v2_instances_match_document():
    doc = json.loads(_doc("result-v016-kimi-2inst.json"))
    src = doc["reports"]["engine"]["instances"]
    m, inst = result_doc.parse_result_document(_doc("result-v016-kimi-2inst.json"), version=2)
    assert [i["instance_id"] for i in inst] == list(src) == ["instance_0", "instance_1"]
    for got, (iid, raw) in zip(inst, src.items(), strict=True):
        assert got == {k: v for k, v in raw.items() if k in result_doc.FLAT_KEYS}, iid
    assert sum(i["completed_requests"] for i in inst) == m["completed_requests"]


def test_v2_one_instance_has_no_instances():
    assert result_doc.parse_result_document(_doc("result-v016-kimi.json"), version=2)[1] == []


def test_v2_drops():
    """`dropped` is omitted when empty; unservable is its own key; an admission rejection leaves
    injected_requests (v0.15.0's meaning: arrivals less admission rejections), so the shed guard sees it."""
    d = json.loads(_doc("result-v016-kimi.json"))
    d["requests"].update(completed=250, dropped={"unservable": 2, "admission_rejected": 4})
    m, _ = result_doc.parse_result_document(json.dumps(d), version=2)
    assert m["dropped_unservable"] == 2
    assert m["injected_requests"] == 252
    assert m["completed_requests"] == 250


@pytest.mark.parametrize("edit, match", [
    (lambda d: d["requests"].pop("queued_at_horizon"), "requests.queued_at_horizon"),
    (lambda d: d["reports"].pop("engine"), "reports.engine"),
    (lambda d: d["requests"].update(dropped={"unservable": "x"}), "requests.dropped.unservable is not a number"),
    (lambda d: d["requests"].update(dropped=[1]), "requests.dropped is not"),
])
def test_v2_bad_document_raises(edit, match):
    d = json.loads(_doc("result-v016-kimi.json"))
    edit(d)
    with pytest.raises(ValueError, match=match):
        result_doc.parse_result_document(json.dumps(d), version=2)


def test_schema_version_must_match():
    with pytest.raises(ValueError, match="schema_version 2.*expected 1"):
        result_doc.parse_result_document(_doc("result-v016-kimi.json"))
    with pytest.raises(ValueError, match="schema_version 1.*expected 2"):
        result_doc.parse_result_document(_doc("result-v015-kimi.json"), version=2)


def test_v2_bad_instance_raises():
    d = json.loads(_doc("result-v016-kimi-2inst.json"))
    d["reports"]["engine"]["instances"]["instance_1"].pop("itl_p99_ms")
    with pytest.raises(ValueError, match="no reports.engine.instances.instance_1.itl_p99_ms"):
        result_doc.parse_result_document(json.dumps(d), version=2)


# --- run_one on v0.16.0 ----------------------------------------------------------------------------

def _fake_blis(monkeypatch, stdout, seen=None):
    def run(cmd, **kw):
        if seen is not None:
            seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout, "")
    monkeypatch.setattr(run_blis.subprocess, "run", run)


def _run(tmp_path, config, catalog="cat", registry="reg"):
    return run_blis.run_one("blis", catalog, registry, str(tmp_path), "kimi-k2.5", "h200", "0.32.0", None, 256,
                            42, None, {**config, "concurrency": 64}, stack=V16)


def test_v016_record(tmp_path, monkeypatch):
    _fake_blis(monkeypatch, _doc("result-v016-kimi.json"))
    rec = _run(tmp_path, CFG, catalog=None)
    assert rec["error"] is None, rec["error"]
    assert rec["metrics"]["injected_requests"] == rec["metrics"]["completed_requests"] == 256
    assert rec["metrics"]["itl_p99_ms"] > 0 and rec["metrics_all_instances"] == []


def test_v016_passes_no_catalog_or_registry(tmp_path, monkeypatch):
    """BLIS prices v0.16.0 on its embedded data; the env's catalog is config-forage's own reading copy."""
    seen = []
    _fake_blis(monkeypatch, _doc("result-v016-kimi.json"), seen)
    _run(tmp_path, CFG)
    assert "--catalog" not in seen[0] and "--registry" not in seen[0]


def test_v016_hides_the_data_env_vars_from_blis(tmp_path, monkeypatch):
    """BLIS v0.16.0 reads BLIS_CATALOG/BLIS_REGISTRY itself (cmd/catalog_root.go:25-26) as overrides of
    its embedded data; config-forage's env uses those names for its reading copy, so BLIS must not see them."""
    envs = []
    monkeypatch.setattr(run_blis.subprocess, "run", lambda cmd, **kw: envs.append(kw.get("env")) or
                        subprocess.CompletedProcess(cmd, 0, _doc("result-v016-kimi.json"), ""))
    monkeypatch.setenv("BLIS_CATALOG", "/x/catalog")
    monkeypatch.setenv("BLIS_REGISTRY", "/x/registry")
    monkeypatch.setenv("PATH", "/bin")
    assert _run(tmp_path, CFG)["error"] is None
    assert envs[0] is not None and "BLIS_CATALOG" not in envs[0] and "BLIS_REGISTRY" not in envs[0]
    assert envs[0]["PATH"] == "/bin"
    run_blis.run_one("blis", "cat", "reg", str(tmp_path), "kimi-k2.5", "h200", "0.32.0", None, 256, 42, None,
                     {**CFG, "concurrency": 64}, stack=STACKS["v0.15.0"])
    assert envs[1] is None                                  # older stacks: the environment as it is


def test_v016_refuses_a_run_not_on_embedded_data(tmp_path, monkeypatch):
    d = json.loads(_doc("result-v016-kimi.json"))
    d["data"]["catalog"]["source"] = "path"
    _fake_blis(monkeypatch, json.dumps(d))
    rec = _run(tmp_path, CFG)
    assert rec["metrics"] == {} and "embedded" in rec["error"]


def test_v016_incomplete_run_is_an_error_row(tmp_path, monkeypatch):
    d = json.loads(_doc("result-v016-kimi.json"))
    d["requests"].update(completed=250, in_flight_at_horizon=6)
    _fake_blis(monkeypatch, json.dumps(d))
    assert _run(tmp_path, CFG)["error"] == "incomplete: injected=256 != completed=250"


def test_v016_shed_guard(tmp_path, monkeypatch):
    _fake_blis(monkeypatch, _doc("result-v016-kimi.json"))
    sc = Scenario("kimi-k2.5", "h200", "chatbot", 8, "0.32.0", stack="v0.16.0")
    r = CachedRunner(None, sc, tmp_path / "cache",
                     run_fn=lambda c, n, s: run_blis.run_one("blis", None, None, str(tmp_path), "kimi-k2.5", "h200",
                                                             "0.32.0", None, n * 4, s, None, {**c, "concurrency": n},
                                                             stack=V16))
    assert r.run(CFG, 80)["error"] == "shed: injected=256 < offered=320"
    assert r.run(CFG, 64)["error"] is None


@pytest.mark.parametrize("key, value", [("scheduler", "fcfs"), ("preemption_policy", "fcfs")])
def test_v016_refuses_deprecated_policy_flags(key, value):
    with pytest.raises(run_blis.ConfigError, match=f"'{key}'.*deprecated in BLIS v0.16.0.*scheduling_policy"):
        run_blis.classify_config({**EP, key: value}, "0.32.0", V16)
    # unchanged on the stacks before it
    assert run_blis.classify_config({**EP, key: value}, "0.32.0", STACKS["v0.15.0"])[2] == {key: value}


def test_v016_cache_keys_differ_from_v015():
    sc15 = Scenario("kimi-k2.5", "h200", "chatbot", 8, "0.32.0", stack="v0.15.0")
    sc16 = Scenario("kimi-k2.5", "h200", "chatbot", 8, "0.32.0", stack="v0.16.0")
    assert runner.cache_key(CFG, 64, 42, sc15) != runner.cache_key(CFG, 64, 42, sc16)


# --- real BLIS --------------------------------------------------------------------------------------

@pytest.mark.blis
@pytest.mark.skipif(not os.environ.get("BLIS_BIN"), reason="BLIS_BIN not set")
def test_v016_reproduces_its_fixture(tmp_path):
    """The run behind result-v016-kimi.json, through CachedRunner, gives the fixture's metrics exactly."""
    from helpers import blis_env_for
    env = blis_env_for("v0.16.0")
    env.scenarios_dir = tmp_path / "scenarios"
    sc = Scenario("kimi-k2.5", "h200", "chatbot", 8, "0.32.0", stack="v0.16.0", seed=42)
    rec = CachedRunner(env, sc, tmp_path / "cache").run({**CFG, "quantization": "fp8", "cache_dtype": "fp8"}, 64)
    want, _ = result_doc.parse_result_document(_doc("result-v016-kimi.json"), version=2)
    assert rec["error"] is None, rec.get("stderr_tail")
    assert rec["metrics"] == want


def test_v016_tables_as_measured():
    """docs/experiments/v016-folds.md: on kimi-k2.5/h200 and minimax-m2.7/h100, every fold but
    all2all_alias is byte-identical; BLIS itself refuses eplb window_size and step_interval."""
    assert V16.measured_folds == frozenset(run_blis.FOLDS) - {"all2all_alias"}
    assert set(V16.inert_block_fields["eplb"]) == {"window_size", "step_interval"}
    for f in ("window_size", "step_interval"):
        with pytest.raises(run_blis.ConfigError, match=f"'{f}'.*v0.16.0.*blis-vllm#76"):
            run_blis.classify_config({**EP, "eplb": {"enable_eplb": True, f: 7}}, "0.32.0", V16)
    ep = {**CFG}
    assert run_blis.canonical_config({**ep, "all2all_backend": "pplx"}, "0.32.0", V16) == {**ep, "all2all_backend": "pplx"}
    assert run_blis.canonical_config({**ep, "long_prefill_token_threshold": 0}, "0.32.0", V16) == ep


@pytest.mark.parametrize("edit", [
    lambda d: d.update(data="embedded"),
    lambda d: d.update(data=["embedded"]),
    lambda d: d["data"].update(catalog="embedded"),
    lambda d: d["data"].update(registry=["embedded"]),
    lambda d: d["data"].update(catalog=7),
])
def test_v016_malformed_data_section_is_an_error_row(tmp_path, monkeypatch, edit):
    """A data section, or a catalog/registry entry, that is not an object is a ValueError (an error
    row from run_one), never an AttributeError escaping the sweep."""
    d = json.loads(_doc("result-v016-kimi.json"))
    edit(d)
    with pytest.raises(ValueError, match=r"data(\.catalog|\.registry)? is not an object"):
        result_doc.check_embedded_data(json.dumps(d))
    _fake_blis(monkeypatch, json.dumps(d))
    rec = _run(tmp_path, CFG)
    assert rec["metrics"] == {} and "is not an object" in rec["error"]
