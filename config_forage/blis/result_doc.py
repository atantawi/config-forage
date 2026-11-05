"""Read a BLIS v0.15.0 Result document into the flat metrics a v0.14.0 run printed.

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


def _instance(iid: str, m) -> dict:
    where = f"engine.instances.{iid}"
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


def parse_result_document(stdout: str) -> tuple[dict, list[dict]]:
    """(metrics, metrics_all_instances) from a BLIS Result document: the cluster's flat metrics with
    instance_id "cluster", and each instance's for a run of more than one. ValueError when stdout is
    not one JSON document of kind Result, or a cluster or instance entry lacks a field the table above
    reads (or holds a non-number there)."""
    try:
        doc = json.loads(stdout)
    except ValueError as e:
        raise ValueError(f"stdout is not a JSON document ({e})") from None
    if not isinstance(doc, dict) or doc.get("kind") != KIND:
        kind = doc.get("kind") if isinstance(doc, dict) else type(doc).__name__
        raise ValueError(f"stdout is not a BLIS Result document (kind {kind!r})")
    metrics = {k: _at(doc, p) for k, p in CLUSTER_FIELDS.items()}
    metrics["instance_id"] = "cluster"
    instances = (doc.get("engine") or {}).get("instances") or {}
    if not isinstance(instances, dict):
        raise ValueError("Result document's engine.instances is not a map of instance id to metrics")  # noqa: TRY004
    per = [_instance(iid, m) for iid, m in instances.items()]   # document order
    return dict(sorted(metrics.items())), per
