"""BLIS stacks: the simulator, catalog and registry versions a space runs against.

A scenario's optional `stack` selects one; absent means DEFAULT_STACK (v0.14.0), whose pins are
exactly the ones every cached run and stored result was made under, so those stay valid."""

from __future__ import annotations

from dataclasses import dataclass, field

V014_COEFFICIENTS = ("cost-model-primitives", "cost-model-collectives", "cost-model-host-overheads",
                     "cost-model-attention", "cost-model-recurrent", "cost-model-memory")

# The no-op engine values run_blis.canonical_config can fold away (see there). A fold is valid on a
# stack only once measured byte-identical on it, so each stack lists the folds it measured.
FOLDS = ("long_prefill_token_threshold_zero",  # long_prefill_token_threshold 0
         "all2all_default",                    # all2all_backend that resolves to the default
         "all2all_alias",                      # an all2all alias becomes the backend it is replaced by
         "eplb_disabled",                      # eplb with enable_eplb false
         "eplb_zero_redundant")                # num_redundant_experts 0 inside an enabled eplb

# eplb fields BLIS v0.14.0 accepts but does not price: refused with the reason (run_blis.engine_block).
V014_INERT_BLOCK_FIELDS = {"eplb": {
    f: "inert in BLIS v0.14.0 (measured identical: glm-5.3-flash/h100/chatbot, 1 x tp4*dp2 + EP, "
       "c=64, eplb on with 32 and with 64 redundant experts, 1 vs 1000000)"
    for f in ("window_size", "step_interval")}}
V015_INERT_BLOCK_FIELDS = {"eplb": {
    f: "inert in BLIS v0.15.0 (measured identical, engine 0.32.0, 1 x tp4*dp2 + EP, c=64, eplb on, 1 vs "
       "1000000: kimi-k2.5/h200/chatbot with 8 and 16 redundant experts, minimax-m2.7/h100/chatbot with "
       "32 and 64)"
    for f in ("window_size", "step_interval")}}

# Re-measured on v0.16.0, engine 0.32.0 (scripts/measure_folds.py; docs/experiments/v016-folds.md), not
# copied: the same as v0.15.0, all2all_alias out (the Result still records the engine's rename warning).
V016_FOLDS = frozenset(FOLDS) - {"all2all_alias"}
# BLIS v0.16.0 refuses these itself when stated away from vLLM's default (blis-vllm v0.32.0-blis.2
# core/vllm/config.go:623-631, issue #76: "not modelled"); refused here first, before a run.
V016_INERT_BLOCK_FIELDS = {"eplb": {
    f: "not modelled in BLIS v0.16.0: blis-vllm v0.32.0-blis.2 refuses a value other than vLLM's default "
       "(blis-vllm#76)"
    for f in ("window_size", "step_interval")}}


@dataclass(frozen=True)
class Stack:
    name: str
    # The last entry is the runner's workload contract, not a BLIS pin: see runner.py.
    pins: tuple[str, ...]
    sim_sha: str
    catalog_tag: str
    registry_tag: str
    engine_version: str
    coefficients: tuple[str, ...]
    # names in FOLDS measured byte-identical on this stack; only these does canonical_config apply
    measured_folds: frozenset[str] = frozenset()
    # {block: {field: reason}}: fields this stack accepts but does not price, refused with the reason
    inert_block_fields: dict = field(default_factory=dict, hash=False)
    # what BLIS prints on stdout: "flat" MetricsOutput objects (run_blis.parse_metrics) or one
    # "result-doc" blis-schemas Result document (result_doc.parse_result_document)
    result_format: str = "flat"
    # whether this stack's kernel prices pipeline parallelism (pp > 1); where it does not, run_blis
    # refuses pp > 1 rather than price it
    pipeline_parallel: bool = False
    # whether BLIS prices with the catalog and registry compiled into it (v0.16.0 on): run_blis then
    # passes no --catalog/--registry and refuses a Result whose data is not "embedded". BlisEnv's
    # catalog is still read here, for workload presets and model graphs: it must be the same data,
    # so for such a stack catalog_tag and registry_tag are the blis-substrate release BLIS embeds.
    embedded_data: bool = False


STACKS: dict[str, Stack] = {
    "v0.14.0": Stack(
        "v0.14.0",
        ("inference-sim v0.14.0 f2ba6bb1c189ed5a56933cfa3d0927eae3836ee9", "kernel v0.4.0",
         "blis-vllm v0.32.0", "catalog 0.3.0", "registry v0.2.0", "workload as token-shape flags"),
        "f2ba6bb1c189ed5a56933cfa3d0927eae3836ee9", "0.3.0", "v0.2.0", "0.29.0", V014_COEFFICIENTS,
        frozenset(FOLDS), V014_INERT_BLOCK_FIELDS),
    "v0.15.0": Stack(
        "v0.15.0",
        ("inference-sim v0.15.0 762f1605ff27cd825d6b2cb2d9e991cd4b864c91", "kernel v0.5.0",
         "blis-vllm v0.32.0-blis.1", "schemas v0.10.0", "router v1.0.0-blis.2", "catalog 0.5.0",
         "registry v0.3.0", "workload as token-shape flags"),
        "762f1605ff27cd825d6b2cb2d9e991cd4b864c91", "0.5.0", "v0.3.0", "0.32.0",
        V014_COEFFICIENTS + ("cost-model-fabric",),
        # re-measured on this stack, engine 0.32.0 (scripts/measure_folds.py; docs/experiments/v015-folds.md):
        # all2all_alias is out -- pplx and naive price as allgather_reducescatter, but the Result
        # document carries the engine's rename warning, so the runs are not byte-identical
        frozenset(FOLDS) - {"all2all_alias"}, V015_INERT_BLOCK_FIELDS, "result-doc",
        # BLIS v0.15.0 prices pp (762f160 "pipelined engines"; blis-vllm v0.32.0-blis.1 runs vLLM's
        # batch queue with async scheduling). It refuses pp with dp > 1 itself (core/vllm/config.go).
        pipeline_parallel=True),
    "v0.16.0": Stack(
        "v0.16.0",
        ("inference-sim v0.16.0 3f4998e28f922611a2d21d5c7ffe18a5272fe114", "substrate v0.13.0",
         "blis-vllm v0.32.0-blis.2", "router v1.0.0-blis.3", "data embedded", "workload as token-shape flags"),
        # catalog and registry: the embedded blis-substrate v0.13.0 (go.mod at 3f4998e); config-forage
        # reads its catalog/ and registry/ from a blis-substrate checkout at that tag
        "3f4998e28f922611a2d21d5c7ffe18a5272fe114", "v0.13.0", "v0.13.0", "0.32.0",
        V014_COEFFICIENTS + ("cost-model-fabric",),
        V016_FOLDS, V016_INERT_BLOCK_FIELDS, "result-doc-v2",
        # blis-vllm v0.32.0-blis.2 still refuses pp with dp > 1 (core/vllm/config.go:379)
        pipeline_parallel=True, embedded_data=True),
}
DEFAULT_STACK = "v0.14.0"


def stack_of(scenario) -> Stack:
    """The stack a scenario names; ValueError for an unknown one."""
    name = getattr(scenario, "stack", None) or DEFAULT_STACK
    try:
        return STACKS[name]
    except KeyError:
        raise ValueError(f"unknown stack {name!r}; known stacks: {sorted(STACKS)}") from None
