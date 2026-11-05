"""The BLIS v0.15.0 Result-document parser (config_forage/blis/result_doc.py).

Fixtures are real BLIS stdout; all but the multidoc-64 one captured through run_blis.run_one:
  result-v015-kimi.json         kimi-k2.5/h200/chatbot, 1 x tp4*dp2 + EP, fp8, c=64, 256 requests, seed 42
  result-v015-kimi-2inst.json   the same with num_instances 2
  result-v015-kimi-multidoc-64.json  kimi-k2.5/h200/multidoc, 64 x tp8 + EP, mnbt 32768, c=64, 256
      requests, seed 42, scenario engine_version 0.29.0 (run_one's scenario file with only that line
      changed, BLIS called directly: run_one refuses 0.29.0 on v0.15.0), BLIS v0.15.0 built GOARCH=amd64
      and run under Rosetta: a 64-instance document for per-instance parsing
  flat-v014-glm-2inst.txt       v0.14.0 flat stdout, glm-5.3-flash/h100/chatbot, 2 x tp8 + EP, c=8
"""

import json
import subprocess
from pathlib import Path

import pytest

from config_forage.blis import result_doc, run_blis
from config_forage.blis.runner import CachedRunner
from config_forage.blis.stacks import STACKS
from config_forage.space import Scenario

DATA = Path(__file__).parent / "data"
FLAT_KEYS = ("completed_requests", "dropped_unservable", "e2e_mean_ms", "e2e_p90_ms", "e2e_p95_ms",
             "e2e_p99_ms", "injected_requests", "itl_mean_ms", "itl_p90_ms", "itl_p95_ms", "itl_p99_ms",
             "kv_allocation_failures", "length_capped_requests", "preemption_count", "responses_per_sec",
             "scheduling_delay_p99_ms", "still_queued", "still_running", "timed_out_requests",
             "tokens_per_sec", "total_input_tokens", "total_output_tokens", "ttft_mean_ms", "ttft_p90_ms",
             "ttft_p95_ms", "ttft_p99_ms", "vllm_estimated_duration_s")

# The fields an instance entry carries: every flat key; kv_allocation_failures is omitempty there.
INSTANCE_KEYS = frozenset(FLAT_KEYS) | {"instance_id"}


def _doc(name):
    return (DATA / name).read_text()


def test_flat_keys_present():
    m, inst = result_doc.parse_result_document(_doc("result-v015-kimi.json"))
    for k in FLAT_KEYS:
        assert isinstance(m.get(k), (int, float)) and not isinstance(m[k], bool), k
    assert m["instance_id"] == "cluster"
    assert set(m) == set(FLAT_KEYS) | {"instance_id"}
    assert inst == []                       # one instance: the document has no engine.instances
    assert m["injected_requests"] == m["completed_requests"] == 256


def test_per_instance():
    m, inst = result_doc.parse_result_document(_doc("result-v015-kimi-2inst.json"))
    assert [i["instance_id"] for i in inst] == ["instance_0", "instance_1"]
    for i in inst:
        for k in ("itl_p99_ms", "ttft_p99_ms", "tokens_per_sec"):
            assert isinstance(i[k], (int, float)), k
    assert sum(i["completed_requests"] for i in inst) == m["completed_requests"]


def _path(doc, dotted):
    for p in dotted.split("."):
        doc = doc[p]
    return doc


# flat key -> the Result field result_doc's table cites, spelled out here independently of the module
CITED = {
    "completed_requests": "requests.completed", "still_queued": "requests.still_queued",
    "still_running": "requests.still_running", "injected_requests": "requests.injected",
    "dropped_unservable": "requests.dropped_unservable", "timed_out_requests": "requests.timed_out",
    "length_capped_requests": "requests.length_capped",
    "total_input_tokens": "engine.cluster.total_input_tokens",
    "total_output_tokens": "engine.cluster.total_output_tokens",
    "vllm_estimated_duration_s": "engine.cluster.simulated_duration_s",
    "responses_per_sec": "engine.cluster.whole_run.responses_per_sec",
    "tokens_per_sec": "engine.cluster.whole_run.tokens_per_sec",
    **{f"{f}_{s}_ms": f"engine.cluster.whole_run.{f}_{s}_ms" for f in ("ttft", "e2e") for s in ("mean", "p90", "p95", "p99")},
    **{f"itl_{s}_ms": f"engine.cluster.whole_run.step_interval_{s}_ms" for s in ("mean", "p90", "p95", "p99")},
    "scheduling_delay_p99_ms": "engine.cluster.scheduling_delay_p99_ms",
    "kv_allocation_failures": "engine.cluster.kv_allocation_failures",
    "preemption_count": "engine.cluster.preemptions",
}


@pytest.mark.parametrize("name", ["result-v015-kimi.json", "result-v015-kimi-2inst.json",
                                  "result-v015-kimi-multidoc-64.json"])
def test_cluster_matches_document(name):
    """Each cluster key is exactly the value at the Result field the mapping table cites."""
    doc = json.loads(_doc(name))
    m, _ = result_doc.parse_result_document(_doc(name))
    assert set(CITED) == set(FLAT_KEYS)
    assert m == {"instance_id": "cluster", **{k: _path(doc, p) for k, p in CITED.items()}}


def test_instances_match_document():
    """All 64 instances, in the document's order (natural: instance_0, instance_1, instance_2, ...):
    each parsed entry is engine.instances.<id> read straight from the fixture, its flat keys only."""
    doc = json.loads(_doc("result-v015-kimi-multidoc-64.json"))
    src = doc["engine"]["instances"]
    _, inst = result_doc.parse_result_document(_doc("result-v015-kimi-multidoc-64.json"))
    assert [i["instance_id"] for i in inst] == list(src) == [f"instance_{n}" for n in range(64)]
    for got, (iid, raw) in zip(inst, src.items(), strict=True):
        assert got == {k: v for k, v in raw.items() if k in INSTANCE_KEYS}, iid
        assert set(got) | {"kv_allocation_failures"} == INSTANCE_KEYS, iid
    for k in ("completed_requests", "injected_requests", "total_input_tokens", "total_output_tokens"):
        assert sum(i[k] for i in inst) == _path(doc, CITED[k]), k


@pytest.mark.parametrize("edit, match", [
    (lambda i: i.pop("itl_p99_ms"), "no engine.instances.instance_1.itl_p99_ms"),
    (lambda i: i.update(tokens_per_sec="fast"), "instance_1.tokens_per_sec is not a number"),
    (lambda i: i.update(completed_requests=True), "instance_1.completed_requests is not a number"),
    (lambda i: i.update(instance_id="instance_9"), "instance_1.instance_id is 'instance_9'"),
])
def test_bad_instance_entry_raises(edit, match):
    d = json.loads(_doc("result-v015-kimi-2inst.json"))
    edit(d["engine"]["instances"]["instance_1"])
    with pytest.raises(ValueError, match=match):
        result_doc.parse_result_document(json.dumps(d))


@pytest.mark.parametrize("stdout, match", [
    ("", "not a JSON document"),
    ("Error: boom", "not a JSON document"),
    ('{"instance_id": "cluster", "tokens_per_sec": 1}', "not a BLIS Result document"),
    ('{"kind": "Scenario"}', "not a BLIS Result document"),
    ("[1, 2]", "not a BLIS Result document"),
])
def test_not_a_result_raises(stdout, match):
    with pytest.raises(ValueError, match=match):
        result_doc.parse_result_document(stdout)


def test_result_missing_a_section_raises():
    d = json.loads(_doc("result-v015-kimi.json"))
    del d["engine"]["cluster"]
    with pytest.raises(ValueError, match="engine.cluster"):
        result_doc.parse_result_document(json.dumps(d))


def _fake_blis(monkeypatch, stdout):
    monkeypatch.setattr(run_blis.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout, ""))


V015_SC = Scenario("kimi-k2.5", "h200", "chatbot", 8, "0.32.0", stack="v0.15.0")
CFG = {"tp": 4, "dp": 2, "enable_expert_parallel": True, "num_instances": 1}


def test_v015_record_has_completion_fields(tmp_path, monkeypatch):
    _fake_blis(monkeypatch, _doc("result-v015-kimi.json"))
    rec = run_blis.run_one("blis", None, None, str(tmp_path), "kimi-k2.5", "h200", "0.32.0", None, 256, 42,
                           None, {**CFG, "concurrency": 64}, stack=STACKS["v0.15.0"])
    assert rec["error"] is None
    assert rec["metrics"]["injected_requests"] == rec["metrics"]["completed_requests"] == 256
    assert rec["metrics"]["itl_p99_ms"] > 0 and rec["metrics_all_instances"] == []

    # _completion_issue sees the request counts: an incomplete run is an error row
    d = json.loads(_doc("result-v015-kimi.json"))
    d["requests"].update(completed=250, still_running=6)
    _fake_blis(monkeypatch, json.dumps(d))
    rec = run_blis.run_one("blis", None, None, str(tmp_path), "kimi-k2.5", "h200", "0.32.0", None, 256, 42,
                           None, {**CFG, "concurrency": 64}, stack=STACKS["v0.15.0"])
    assert rec["error"] == "incomplete: injected=256 != completed=250"

    # _shed_guard sees injected_requests: 256 injected of 320 offered (c=80) is shed
    _fake_blis(monkeypatch, _doc("result-v015-kimi.json"))
    runner = CachedRunner(None, V015_SC, tmp_path / "cache",
                          run_fn=lambda c, n, s: run_blis.run_one(
                              "blis", None, None, str(tmp_path), "kimi-k2.5", "h200", "0.32.0", None,
                              n * 4, s, None, {**c, "concurrency": n}, stack=STACKS["v0.15.0"]))
    assert runner.run(CFG, 80)["error"] == "shed: injected=256 < offered=320"
    assert runner.run(CFG, 64)["error"] is None


def test_v015_unparseable_stdout_is_an_error_row(tmp_path, monkeypatch):
    _fake_blis(monkeypatch, '{"instance_id": "cluster", "tokens_per_sec": 1}')
    rec = run_blis.run_one("blis", None, None, str(tmp_path), "kimi-k2.5", "h200", "0.32.0", None, 256, 42,
                           None, {**CFG, "concurrency": 64}, stack=STACKS["v0.15.0"])
    assert rec["returncode"] == 0 and rec["metrics"] == {} and rec["metrics_all_instances"] == []
    assert rec["error"].startswith("rc=0 but stdout is not a BLIS Result document")


def test_stack_result_formats():
    assert STACKS["v0.14.0"].result_format == "flat"
    assert STACKS["v0.15.0"].result_format == "result-doc"


def test_v014_parse_unchanged(tmp_path, monkeypatch):
    """v0.14.0 still reads the flat MetricsOutput objects, through parse_metrics/parse_all_instances."""
    stdout = _doc("flat-v014-glm-2inst.txt")
    _fake_blis(monkeypatch, stdout)
    rec = run_blis.run_one("blis", None, None, str(tmp_path), "glm-5.3-flash", "h100", "0.29.0", None, 80, 42,
                           None, {"tp": 8, "enable_expert_parallel": True, "num_instances": 2, "concurrency": 8},
                           stack=STACKS["v0.14.0"])
    assert rec["metrics"] == run_blis.parse_metrics(stdout)
    assert rec["metrics_all_instances"] == run_blis.parse_all_instances(stdout)
    assert json.dumps(rec["metrics"], sort_keys=True) == json.dumps(
        next(o for o in run_blis._iter_json_objects(stdout) if o.get("instance_id") == "cluster"), sort_keys=True)
    assert [i["instance_id"] for i in rec["metrics_all_instances"]] == ["instance_0", "instance_1"]
    assert rec["error"] is None
    # and the flat output is not a Result document
    with pytest.raises(ValueError):
        result_doc.parse_result_document(stdout)


def test_v014_and_v015_records_share_their_key_set():
    """A v0.15.0 cluster entry has the v0.14.0 flat cluster entry's keys (v0.14.0 omits
    kv_allocation_failures when zero; omitempty in MetricsOutput), and so do the instances."""
    flat = _doc("flat-v014-glm-2inst.txt")
    v14, v14_inst = run_blis.parse_metrics(flat), run_blis.parse_all_instances(flat)
    v15, v15_inst = result_doc.parse_result_document(_doc("result-v015-kimi-2inst.json"))
    assert set(v14) - {"kv_allocation_failures"} <= set(v15)
    assert set(v15) - set(v14) <= {"kv_allocation_failures"}
    assert {k for i in v14_inst for k in i} == {k for i in v15_inst for k in i}
