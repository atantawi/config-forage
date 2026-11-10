"""Read a BLIS Result document (v0.15.0: schema 1; v0.16.0: schema 2) into the flat metrics a v0.14.0 run printed.

From v0.15.0, `blis run` prints one blis-schemas Result (`kind: Result`) on stdout and nothing else
(inference-sim docs/guide/results.md:3-8). Scoring here (run_blis._completion_issue,
CachedRunner._shed_guard, the probe's ITL SLO, the ladder's worst instance) reads the flat
`MetricsOutput` keys v0.14.0 printed. parse_result_document gives those keys, with their v0.14.0
meaning, read from where the Result put each one.

Window or whole run. The Result's top-level `metrics` are over the measured window and its `itl_*`
are per token (results.md:43-64). v0.14.0's flat keys are the whole-run figures and its `itl_*` are
per decode step: the flat object was `aggregated.BuildOutput("cluster")` (blis-stack-v014
inference-sim cmd/root.go:1475; its sim/metrics.go:256-326). v0.15.0's BuildOutput is the same function
(sim/metrics.go:259-329: ITL over m.AllITLs, the step series, at :301-306; tokens and responses over the
simulated run at :316-318). The Result keeps that object's figures under `engine.cluster.whole_run`,
built from `m.BuildOutput("cluster")` (cmd/result_inputs.go:296-310),
so every latency, `tokens_per_sec` and `responses_per_sec` is read there, not from `metrics`. That
keeps the v0.14.0 meaning; the windowed `metrics.output_tok_s` and `metrics.itl_p99_ms` would not.

Source paths: R = ~/go/pkg/mod/github.com/inference-sim/blis-schemas@v0.10.0/spec/result/result.go;
I = blis-stack-v015/inference-sim/cmd/result_inputs.go; D = inference-sim/docs/guide/results.md
("Where each MetricsOutput field went"); M = inference-sim/sim/metrics.go.

| flat key                         | Result field                                  | source                     |
|----------------------------------|-----------------------------------------------|----------------------------|
| instance_id                      | "cluster" (dropped from the Result)           | D:75                       |
| completed_requests               | requests.completed                            | D:76, R:182, I:211         |
| still_queued                     | requests.still_queued                         | D:77, R:183, I:212         |
| still_running                    | requests.still_running                        | D:78, R:184, I:213         |
| injected_requests                | requests.injected                             | D:79, R:180, I:210         |
| dropped_unservable               | requests.dropped_unservable                   | D:80, R:185, I:214         |
| timed_out_requests               | requests.timed_out                            | D:81, R:186, I:215         |
| length_capped_requests           | requests.length_capped                        | D:82, R:198, I:222         |
| total_input_tokens               | engine.cluster.total_input_tokens             | D:83, I:284                |
| total_output_tokens              | engine.cluster.total_output_tokens            | D:84, I:286                |
| vllm_estimated_duration_s        | engine.cluster.simulated_duration_s           | D:85, I:287 (= M:260)      |
| responses_per_sec                | engine.cluster.whole_run.responses_per_sec    | D:86, I:309                |
| tokens_per_sec                   | engine.cluster.whole_run.tokens_per_sec       | D:87, I:309                |
| ttft_{mean,p90,p95,p99}_ms       | engine.cluster.whole_run.ttft_*_ms            | D:88, I:305                |
| e2e_{mean,p90,p95,p99}_ms        | engine.cluster.whole_run.e2e_*_ms             | D:89, I:306                |
| itl_{mean,p90,p95,p99}_ms        | engine.cluster.whole_run.step_interval_*_ms   | D:90, I:307-308            |
| scheduling_delay_p99_ms          | engine.cluster.scheduling_delay_p99_ms        | D:91, I:269                |
| kv_allocation_failures           | engine.cluster.kv_allocation_failures         | D:92, I:268                |
| preemption_count                 | engine.cluster.preemptions                    | D:93, I:267                |

`requests.*` and the `engine.cluster` counts are the aggregated metrics' own counters, as the flat
object's were (I:204-224; M:262-276). One difference: v0.14.0's injected_requests was the sum of
completed, still_queued, still_running, dropped_unservable and timed_out (M:267), while
`requests.injected` is arrivals less admission rejections (I:209), which INV-1 makes that sum plus
the router's buckets (routing_rejections, gateway_*; R:201-207). They agree when the router drops
nothing, as in every run here; the Result's is taken, as D:79 maps it.

Per instance. A run of more than one instance keys each instance's own `MetricsOutput`
(`inst.Metrics().BuildOutput(id)`, the object v0.14.0 printed per instance) by id under
`engine.instances` (D:75; I:330-342). Those are already flat and are taken as they are, restricted to
the keys above, in document order (natural: instance_0, instance_1, instance_2, ...). Each must carry
every flat key as a number, and its own instance_id must be its key, else ValueError, as for the cluster;
`kv_allocation_failures` alone may be absent: it is omitempty there (sim/metrics_utils.go:115), as on v0.14.0.
A one-instance run has no `engine.instances` (I:334-336): `metrics_all_instances` is [], as on v0.14.0.

SCHEMA_2 (BLIS v0.16.0, 3f4998e). The document is blis-substrate v0.13.0's contracts/spec/result
(`schema_version: 2`). Each flat key's home is the v0.16.0 table "Where each MetricsOutput field went"
(D2 = inference-sim v0.16.0 docs/guide/results.md:121-141; R2 = blis-substrate@v0.13.0
contracts/spec/result/result.go; I2 = inference-sim v0.16.0 cmd/result_inputs.go). What changed:

| flat key           | schema 1                     | schema 2                                                      |
|--------------------|------------------------------|---------------------------------------------------------------|
| still_queued       | requests.still_queued        | requests.queued_at_horizon (D2:125)                           |
| still_running      | requests.still_running       | requests.in_flight_at_horizon (D2:126)                        |
| injected_requests  | requests.injected            | requests.generated less requests.dropped.admission_rejected   |
| dropped_unservable | requests.dropped_unservable  | requests.dropped.unservable, 0 when absent (D2:128)           |
| engine.cluster.*   | engine.cluster               | reports.engine.cluster, same keys and whole_run (D2:131-141)  |
| per instance       | engine.instances             | reports.engine.instances, the same flat objects (D2:123)      |

`requests.dropped` is a map of reason to count, omitted when empty (R2:489; I2:239-246 adds only
reasons with a count). injected_requests: D2:127 maps it to `generated`, but `generated` counts every
request the generator made, admission-rejected ones included (the fates sum to it, R2:541-548).
v0.15.0's `requests.injected` was arrivals less admission rejections, and CachedRunner._shed_guard
relies on that meaning: an admission-shedding run must show fewer injected than offered. So
injected_requests is `generated` less `dropped.admission_rejected`; with no admission rejections, as
in every run here, it is `generated`, D2's mapping. The other drop reasons (routing, gateway,
unservable) stay in it, as they did in v0.15.0's injected.
"""

from __future__ import annotations

import json

KIND = "Result"

# flat key -> path in the Result document (see the table above)
CLUSTER_FIELDS: dict[str, tuple[str, ...]] = {
    "completed_requests": ("requests", "completed"),
    "still_queued": ("requests", "still_queued"),
    "still_running": ("requests", "still_running"),
    "injected_requests": ("requests", "injected"),
    "dropped_unservable": ("requests", "dropped_unservable"),
    "timed_out_requests": ("requests", "timed_out"),
    "length_capped_requests": ("requests", "length_capped"),
    "total_input_tokens": ("engine", "cluster", "total_input_tokens"),
    "total_output_tokens": ("engine", "cluster", "total_output_tokens"),
    "vllm_estimated_duration_s": ("engine", "cluster", "simulated_duration_s"),
    "responses_per_sec": ("engine", "cluster", "whole_run", "responses_per_sec"),
    "tokens_per_sec": ("engine", "cluster", "whole_run", "tokens_per_sec"),
    **{f"{fam}_{s}_ms": ("engine", "cluster", "whole_run", f"{fam}_{s}_ms")
       for fam in ("ttft", "e2e") for s in ("mean", "p90", "p95", "p99")},
    **{f"itl_{s}_ms": ("engine", "cluster", "whole_run", f"step_interval_{s}_ms")
       for s in ("mean", "p90", "p95", "p99")},
    "scheduling_delay_p99_ms": ("engine", "cluster", "scheduling_delay_p99_ms"),
    "kv_allocation_failures": ("engine", "cluster", "kv_allocation_failures"),
    "preemption_count": ("engine", "cluster", "preemptions"),
}
FLAT_KEYS = frozenset(CLUSTER_FIELDS) | {"instance_id"}

# schema 2 (v0.16.0): flat key -> path, for the keys with one home (see SCHEMA_2 in the docstring);
# injected_requests and dropped_unservable are derived in _cluster_v2
_V2_MOVED = {"still_queued": ("requests", "queued_at_horizon"),
             "still_running": ("requests", "in_flight_at_horizon")}
CLUSTER_FIELDS_V2: dict[str, tuple[str, ...]] = {
    k: _V2_MOVED.get(k, ("reports", *p) if p[0] == "engine" else p)
    for k, p in CLUSTER_FIELDS.items() if k not in ("injected_requests", "dropped_unservable")}
# an instance's MetricsOutput omits kv_allocation_failures when zero (omitempty)
INSTANCE_OPTIONAL = frozenset({"kv_allocation_failures"})


def _number(v, where: str):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"Result document's {where} is not a number: {v!r}")  # noqa: TRY004 (callers catch ValueError)
    return v


def _at(doc: dict, path: tuple[str, ...]):
    v = doc
    for i, p in enumerate(path):
        if not isinstance(v, dict) or p not in v:
            raise ValueError(f"Result document has no {'.'.join(path[:i + 1])}")
        v = v[p]
    return _number(v, ".".join(path))


def _instance(iid: str, m, where: str) -> dict:
    where = f"{where}.{iid}"
    if not isinstance(m, dict):
        raise ValueError(f"Result document's {where} is not an object")  # noqa: TRY004
    if m.get("instance_id") != iid:
        raise ValueError(f"Result document's {where}.instance_id is {m.get('instance_id')!r}, not {iid!r}")
    out = {"instance_id": iid}
    for k in sorted(CLUSTER_FIELDS):
        if k in m:
            out[k] = _number(m[k], f"{where}.{k}")
        elif k not in INSTANCE_OPTIONAL:
            raise ValueError(f"Result document has no {where}.{k}")
    return dict(sorted(out.items()))


def _v2_drops(doc: dict) -> dict:
    """requests.dropped (by reason; omitted when empty), each count checked to be a number."""
    dropped = doc["requests"].get("dropped", {}) if isinstance(doc.get("requests"), dict) else {}
    if not isinstance(dropped, dict):
        raise ValueError("Result document's requests.dropped is not a map of reason to count")  # noqa: TRY004
    return {r: _number(n, f"requests.dropped.{r}") for r, n in dropped.items()}


def _cluster_v2(doc: dict) -> dict:
    """The cluster's flat metrics from a schema 2 Result (see SCHEMA_2 above)."""
    metrics = {k: _at(doc, p) for k, p in CLUSTER_FIELDS_V2.items()}
    _at(doc, ("requests", "generated"))
    drops = _v2_drops(doc)
    metrics["dropped_unservable"] = drops.get("unservable", 0)
    metrics["injected_requests"] = doc["requests"]["generated"] - drops.get("admission_rejected", 0)
    return metrics


def parse_result_document(stdout: str, version: int = 1) -> tuple[dict, list[dict]]:
    """(metrics, metrics_all_instances) from a BLIS Result document of schema `version` (1: v0.15.0,
    2: v0.16.0): the cluster's flat metrics with instance_id "cluster", and each instance's for a run
    of more than one. ValueError when stdout is not one JSON document of kind Result at that
    schema_version, or a cluster or instance entry lacks a field the tables above read (or holds a
    non-number there)."""
    try:
        doc = json.loads(stdout)
    except ValueError as e:
        raise ValueError(f"stdout is not a JSON document ({e})") from None
    if not isinstance(doc, dict) or doc.get("kind") != KIND:
        kind = doc.get("kind") if isinstance(doc, dict) else type(doc).__name__
        raise ValueError(f"stdout is not a BLIS Result document (kind {kind!r})")
    if doc.get("schema_version") != version:
        raise ValueError(f"Result document is schema_version {doc.get('schema_version')!r}, expected {version}")
    if version == 2:
        metrics, engine, where = _cluster_v2(doc), (doc.get("reports") or {}).get("engine") or {}, \
            "reports.engine.instances"
    else:
        metrics, engine, where = {k: _at(doc, p) for k, p in CLUSTER_FIELDS.items()}, doc.get("engine") or {}, \
            "engine.instances"
    metrics["instance_id"] = "cluster"
    instances = engine.get("instances") or {}
    if not isinstance(instances, dict):
        raise ValueError(f"Result document's {where} is not a map of instance id to metrics")  # noqa: TRY004
    per = [_instance(iid, m, where) for iid, m in instances.items()]   # document order
    return dict(sorted(metrics.items())), per


def check_embedded_data(stdout: str) -> None:
    """ValueError unless the Result says BLIS read both its catalog and its registry from the copies
    embedded in it (`data.catalog.source` and `data.registry.source` "embedded"; inference-sim v0.16.0
    docs/guide/results.md:29): a stack with embedded data is priced on those, and its pins name no other."""
    data = json.loads(stdout).get("data") or {}
    for what in ("catalog", "registry"):
        src = (data.get(what) or {}).get("source") if isinstance(data, dict) else None
        if src != "embedded":
            raise ValueError(f"Result document's data.{what}.source is {src!r}, not 'embedded'")
