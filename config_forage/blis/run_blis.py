#!/usr/bin/env python3
# Vendored from inference-sim/blis-config-search (Apache-2.0) at 1a6dc6fa89b7c3569b409b9b5c92143267dd80ad,
# then modified for BLIS v0.14.0 and v0.15.0 (see NOTICE; per-stack tables in config_forage.blis.stacks): routing goes through the embedded llm-d router, which
# the scenario's `router:` block configures with an EndpointPickerConfig (epp_config, or the flat router
# knobs of config_forage.router); the BLIS-native routing/admission flags were removed in v0.12.0. DP
# load balancing is a pool parallel field.
# run_blis.py -- the ONE place config-search invokes BLIS and records a single run.
#
#   python3 -I scripts/run_blis.py ...        (importable: sweep_pool.py reuses run_one())
#
# BLIS is driven on the **blis-latency-kernel** backend. The kernel reads the deployment (model,
# hardware, tp/pp/dp/ep, engine knobs) from a committed SCENARIO and REFUSES
# --model/--hardware/--tp/--dp/--enable-expert-parallel on the CLI. So a candidate's per-instance
# topology is written into a generated Scenario+Deployment YAML; only workload, concurrency, instance
# counts, scheduler/preemption policy, and seed stay on the CLI; the router config is a scenario block.
# The workload goes on the CLI as its token-shape flags, never as --workload: closed-loop BLIS ignores
# --workload (config_forage.blis.workloads).
#
# Every candidate key is classified to its scenario/CLI slot; UNKNOWN or NOT-YET-SUPPORTED keys are
# REJECTED (recorded as an error row), never silently dropped -- else two candidates could differ in
# the ledger yet run the identical command.
#
# Pinned stacks (config_forage.blis.stacks): every per-stack fact here -- coefficients, the folds
# canonical_config applies, the inert fields engine_block refuses -- is taken from the run's Stack, and
# defaults to v0.14.0 (BLIS @ f2ba6bb, kernel v0.4.0, blis-vllm v0.32.0 in go.mod, blis-catalog 0.3.0,
# blis-registry v0.2.0), so a caller that names no stack runs exactly as before stacks existed.
import argparse, hashlib, json, os, subprocess, sys, tempfile, time

from config_forage import router as router_knobs
from config_forage.blis import result_doc, workloads
from config_forage.blis.stacks import DEFAULT_STACK, FOLDS, STACKS  # noqa: F401  (FOLDS re-exported)

V014 = STACKS[DEFAULT_STACK]
COEFFICIENTS = list(V014.coefficients)      # v0.14.0's; a stack's own are Stack.coefficients

# --- candidate-key classification (allowlist; anything else is rejected, never silently ignored) ---
SCENARIO_PARALLEL = ("tp", "pp", "dp", "enable_expert_parallel",       # -> scenario Deployment.parallel
                     "data_parallel_load_balancing")
# -> the scenario's `router.endpoint_picker_config`: an EndpointPickerConfig (a dict) written next to it
SCENARIO_ROUTER = ("epp_config",)
SCENARIO_ENGINE = ("quantization", "cache_dtype", "block_size", "max_num_batched_tokens",
                   "max_num_seqs", "max_model_len", "cudagraph_mode", "gpu_memory_utilization",
                   "all2all_backend", "long_prefill_token_threshold")
# -> a sub-block of the scenario's engine section: {block: {field: type}}. Field names and types are
# blis-vllm v0.32.0's engineconfig (EPLB; v0.32.0-blis.1 rules/engineconfig/engine.go:314-319 has the
# same four); BLIS refuses an unknown field, so refuse it here first,
# before a run. enable_eplb must be stated: a block without it is EPLB off, the same run as no block.
# With EPLB on, the engine requires the EP width to divide experts + num_redundant_experts
# (blis-vllm rules "expert-divisibility-under-eplb"), and `blis run` does not enforce it: an
# indivisible layout runs and is priced. classify_config does not know the model, so run_one checks
# it against the catalog (check_eplb_layout); classify_config refuses EPLB without expert parallelism.
SCENARIO_ENGINE_BLOCKS = {"eplb": {"enable_eplb": bool, "num_redundant_experts": int,
                                   "window_size": int, "step_interval": int}}
REQUIRED_BLOCK_FIELDS = {"eplb": ("enable_eplb",)}
# Block fields a stack accepts but does not price -> refused with the reason, as UNSUPPORTED is
# (Stack.inert_block_fields; measured per stack). This is v0.14.0's: window_size and step_interval.
UNSUPPORTED_BLOCK_FIELDS = V014.inert_block_fields
# Engine-release facts BLIS prices but `blis run` does not validate: a value outside the legal set is
# priced anyway (an unrecognized all2all_backend gets the wrong volume basis), so it is refused here.
# 0.29.0: transcribed from blis-vllm v0.32.0 rules/v0_29/pack.go (All2AllBackends, DefaultAll2AllBackend,
# All2AllBackendRenames), as pinned by inference-sim v0.14.0. 0.32.0: from blis-vllm v0.32.0-blis.1
# rules/v0_32/pack.go:131-135 (All2AllBackends: 0.31.0's thirteen and passthrough), :267
# (DefaultAll2AllBackend) and :268-269 (All2AllBackendRenames), as pinned by inference-sim v0.15.0.
# Add a release by adding its entry.
#   aliases: names the engine accepts but replaces by another backend; measured byte-identical to
#   that backend (glm-5.3-flash/h100/chatbot, 1 x tp4*dp2 + EP, c=64: pplx and naive both priced
#   exactly as allgather_reducescatter), so canonical_config rewrites them and they share its runs.
ENGINE_RELEASES = {
    "0.29.0": {
        "all2all_backends": ("naive", "allgather_reducescatter", "pplx", "deepep_high_throughput",
                             "deepep_low_latency", "deepep_v2", "flashinfer_all2allv",
                             "flashinfer_nvlink_one_sided", "flashinfer_nvlink_two_sided",
                             "mori_high_throughput", "mori_low_latency", "nixl_ep"),
        "all2all_default": "allgather_reducescatter",
        "all2all_aliases": {"pplx": "allgather_reducescatter", "naive": "allgather_reducescatter"},
    },
    # aliases here: the engine's renames, folded only on a stack that measured them (Stack.measured_folds)
    "0.32.0": {
        "all2all_backends": ("naive", "allgather_reducescatter", "pplx", "deepep_high_throughput",
                             "deepep_low_latency", "deepep_v2", "flashinfer_all2allv",
                             "flashinfer_nvlink_one_sided", "flashinfer_nvlink_two_sided",
                             "mori_high_throughput", "mori_low_latency", "moonep", "nixl_ep",
                             "passthrough"),
        "all2all_default": "allgather_reducescatter",
        "all2all_aliases": {"pplx": "allgather_reducescatter", "naive": "allgather_reducescatter"},
    },
}
DEFAULT_ENGINE_VERSION = "0.29.0"   # the CLI's --engine-version default
CLI_FLAGS = {"num_instances": "--num-instances", "concurrency": "--concurrency",
             "scheduler": "--scheduler", "preemption_policy": "--preemption-policy"}
PARALLEL_DEFAULTS = {"tp": 1, "pp": 1, "dp": 1, "enable_expert_parallel": False}
GPUS_PER_NODE = 8   # per node (H100/H200 HGX); a footprint beyond this spans whole nodes over a fabric
# Cross-node cost resolves from the fabric (a catalog networks/ entry), required once nodes > 1.
DEFAULT_FABRIC = "ib-400g"   # InfiniBand NDR 400 Gb/s; the standard H100 multi-node interconnect
# NON-PRECISION engine defaults only. quantization/cache_dtype are intentionally NOT defaulted: an
# empty precision lets the checkpoint format govern (e.g. qwen3-14b stays BF16), so forcing FP8 here
# would silently mis-price every non-FP8 model. `block_size` is also absent here because its right
# value depends on the MODEL -- see default_block_size(), which fills it in per run.
ENGINE_DEFAULTS = {"max_num_batched_tokens": 8192, "max_num_seqs": 256,
                   "max_model_len": 32768, "cudagraph_mode": "PIECEWISE", "gpu_memory_utilization": 0.9}
# A DSA sparse-MLA model's indexer runs on 64-token kernel blocks, so a stated block_size must be a
# multiple of 64 or the engine refuses the layout at startup (blis-latency-kernel sparse_mla.go,
# `dsaBlockSize`). vLLM v0.31.0 picks 64 for these models for the same reason.
DSA_BLOCK_SIZE = 64
DEFAULT_BLOCK_SIZE = 16     # vLLM's own default for everything else
# Deliberately unsupported in this runner -> fail loudly with the reason (do not silently drop).
UNSUPPORTED = {
    "prefill_instances": "PD disaggregation is not supported yet (single colocated pool only)",
    "decode_instances": "PD disaggregation is not supported yet (single colocated pool only)",
    "prefill_tp": "per-role PD TP is not supported yet (single colocated pool only)",
    "decode_tp": "per-role PD TP is not supported yet (single colocated pool only)",
    "num_speculative_tokens": "speculative decoding is not yet validated on the kernel backend",
    "speculative_acceptance_rate": "speculative decoding is not yet validated on the kernel backend",
    "speculative_method": "speculative decoding is not yet validated on the kernel backend",
    "routing_policy": "removed in BLIS v0.12.0: route with an EndpointPickerConfig (epp_config)",
    "routing_scorers": "removed in BLIS v0.12.0: route with an EndpointPickerConfig (epp_config)",
    "admission_policy": "removed in BLIS v0.12.0: admit with an EndpointPickerConfig (epp_config)",
}


class ConfigError(ValueError):
    pass


def check_stack_engine(engine_version, stack):
    """Refuse an engine_version other than the stack's own: a stack's tables (folds, inert fields)
    were measured at that one engine release, so they say nothing about another."""
    if engine_version != stack.engine_version:
        raise ConfigError(f"engine_version {engine_version!r} is not stack {stack.name}'s "
                          f"{stack.engine_version!r}; a stack runs one engine release")


def canonical_config(config, engine_version=DEFAULT_ENGINE_VERSION, stack=V014):
    """`config` with its measured no-op engine values made absent, so a no-op spelling shares the
    default's cache key and scenario file instead of buying a duplicate run. Each fold (FOLDS) applies
    only on a stack that lists it in Stack.measured_folds: it was measured byte-identical to leaving the
    value out on that stack (v0.14.0: glm-5.3-flash/h100/chatbot, 1 x tp4*dp2 + EP, c=64; v0.15.0:
    docs/experiments/v015-folds.md), and each is also the engine's own zero value or default:
      - long_prefill_token_threshold_zero: long_prefill_token_threshold 0 (uncapped prefills, the
        engine default);
      - all2all_default: all2all_backend that resolves (through the release's aliases, when
        all2all_alias is measured too) to the release's default;
      - all2all_alias: an all2all alias otherwise becomes the backend it is replaced by;
      - eplb_disabled: eplb with enable_eplb false (the whole block: EPLB off, whatever else it states);
      - eplb_zero_redundant: num_redundant_experts 0 inside an enabled eplb.
    `{enable_eplb: true}` is NOT dropped: it measured identical only at an EP width dividing the
    expert count, and the engine prices an indivisible width differently with EPLB on. Malformed
    values are left as they are, for classify_config to refuse. ConfigError when engine_version is
    not the stack's (check_stack_engine)."""
    check_stack_engine(engine_version, stack)
    out, folds = dict(config), stack.measured_folds
    if ("long_prefill_token_threshold_zero" in folds and out.get("long_prefill_token_threshold") == 0
            and type(out["long_prefill_token_threshold"]) is int):
        del out["long_prefill_token_threshold"]
    rel = ENGINE_RELEASES.get(engine_version)
    a2a = out.get("all2all_backend")
    if rel and isinstance(a2a, str):
        if "all2all_alias" in folds:
            a2a = out["all2all_backend"] = rel["all2all_aliases"].get(a2a, a2a)
        if "all2all_default" in folds and a2a == rel["all2all_default"]:
            del out["all2all_backend"]
    eplb = out.get("eplb")
    try:
        engine_block("eplb", eplb, stack) if "eplb" in out else None
    except ConfigError:
        eplb = None                                 # malformed: left for classify_config to refuse
    if isinstance(eplb, dict):
        if "eplb_disabled" in folds and eplb.get("enable_eplb") is False:
            del out["eplb"]
        elif ("eplb_zero_redundant" in folds and eplb.get("num_redundant_experts") == 0
              and type(eplb["num_redundant_experts"]) is int):
            out["eplb"] = {f: x for f, x in eplb.items() if f != "num_redundant_experts"}
    return out


def check_engine_values(engine, engine_version):
    """Refuse an engine value BLIS would price without validating (see ENGINE_RELEASES)."""
    if "all2all_backend" in engine:
        v = engine["all2all_backend"]
        rel = ENGINE_RELEASES.get(engine_version)
        if rel is None:
            raise ConfigError(f"'all2all_backend': no legal backend list for engine_version "
                              f"{engine_version!r}; known = {sorted(ENGINE_RELEASES)}")
        if not isinstance(v, str) or v not in rel["all2all_backends"]:
            raise ConfigError(f"'all2all_backend': {v!r} is not accepted by engine {engine_version}; "
                              f"legal = {sorted(rel['all2all_backends'])}")
    if "long_prefill_token_threshold" in engine:
        v, cap = engine["long_prefill_token_threshold"], engine["max_model_len"]
        if type(v) is not int or v < 0:
            raise ConfigError(f"'long_prefill_token_threshold': must be a non-negative integer, got {v!r}")
        # the engine refuses one above max_model_len at startup (vllm/config/scheduler.py
        # verify_max_model_len); `blis run` does not, and would price it
        if isinstance(cap, int) and v > cap:
            raise ConfigError(f"'long_prefill_token_threshold': {v} exceeds max_model_len {cap}; "
                              f"the engine refuses it at startup")


def classify_config(config, engine_version=DEFAULT_ENGINE_VERSION, stack=V014):
    """Split a candidate into (parallel, engine, cli, router); raise ConfigError on unknown/unsupported
    keys, illegal engine values for `engine_version`, or an engine_version not `stack`'s. Router knobs (config_forage.router) are
    translated to epp_config first, and measured no-op engine values dropped (canonical_config).
    router is {} or {"epp_config": <EndpointPickerConfig dict>}."""
    check_stack_engine(engine_version, stack)
    try:
        config = router_knobs.to_blis(config)
    except ValueError as e:
        raise ConfigError(str(e)) from None
    parallel, engine, cli, router = dict(PARALLEL_DEFAULTS), dict(ENGINE_DEFAULTS), {}, {}
    raw_engine = {k: config[k] for k in ("all2all_backend", "long_prefill_token_threshold") if k in config}
    check_engine_values({"max_model_len": config.get("max_model_len", ENGINE_DEFAULTS["max_model_len"]),
                         **raw_engine}, engine_version)
    if "eplb" in config:
        engine_block("eplb", config["eplb"], stack)  # refuse a malformed block before it is normalised
    config = canonical_config(config, engine_version, stack)
    for k, v in config.items():
        if k in SCENARIO_PARALLEL:
            parallel[k] = v
        elif k in SCENARIO_ROUTER:
            if not isinstance(v, dict) or v.get("kind") != "EndpointPickerConfig":
                raise ConfigError(f"{k!r}: must be an EndpointPickerConfig mapping")
            router[k] = v
        elif k in SCENARIO_ENGINE:
            engine[k] = v
        elif k in SCENARIO_ENGINE_BLOCKS:
            engine[k] = engine_block(k, v, stack)
        elif k in CLI_FLAGS:
            cli[k] = v
        elif k in UNSUPPORTED:
            raise ConfigError(f"{k!r}: {UNSUPPORTED[k]}")
        else:
            supported = (sorted(SCENARIO_PARALLEL) + sorted(SCENARIO_ROUTER) + sorted(SCENARIO_ENGINE)
                         + sorted(SCENARIO_ENGINE_BLOCKS) + sorted(CLI_FLAGS))
            raise ConfigError(f"{k!r}: unknown key; supported = {supported}")
    pp = parallel["pp"]
    if type(pp) is not int or pp < 1:
        raise ConfigError(f"'pp': must be a positive integer, got {pp!r}")
    if pp != 1 and not stack.pipeline_parallel:
        raise ConfigError(f"pp>1: stack {stack.name}'s kernel does not price pipeline parallelism")
    if pp != 1 and parallel["dp"] != 1:
        # BLIS refuses it at startup (blis-vllm v0.32.0-blis.1 core/vllm/config.go: the DP group is
        # diff-tested on one pipeline stage per rank); refused here first, before a run
        raise ConfigError(f"pp {pp} with dp {parallel['dp']}: BLIS does not build pipeline parallelism "
                          f"inside data-parallel ranks")
    fp = footprint_gpus(parallel)
    # A multi-node single instance IS supported: the generated scenario names a fabric once the
    # pool spans nodes (see scenario_yaml), and BLIS prices the cross-node collectives from it.
    # This matters because the largest MoE models cannot fit in one node at all -- kimi-k3 needs
    # 332.3 GiB per rank at tp=8 on a 141 GiB device and only fits from tp=32 (4 nodes) up.
    # What is still refused is a footprint that STRADDLES a node boundary: pools are allocated in
    # whole nodes, so the remainder would sit idle and the cross-node term would be mispriced.
    if fp > GPUS_PER_NODE and fp % GPUS_PER_NODE:
        raise ConfigError(
            f"per-instance footprint tp*pp*dp={fp} spans {fp / GPUS_PER_NODE:.2g} nodes: a "
            f"multi-node instance must fill whole {GPUS_PER_NODE}-GPU nodes (use a footprint that "
            f"is a multiple of {GPUS_PER_NODE}, or scale out with num_instances instead)")
    if engine.get("eplb", {}).get("enable_eplb") and parallel["enable_expert_parallel"] is not True:
        # BLIS prices it (EP width 1 with EPLB on: 32 redundant experts measured 1.7% slower than
        # none, glm-5.3-flash tp4*dp2), but the engine rebalances experts only across EP ranks
        raise ConfigError("'eplb': enable_eplb needs enable_expert_parallel")
    lb = parallel.get("data_parallel_load_balancing")
    if lb is not None and parallel["dp"] == 1:
        # without DP there is nothing to balance; stating it would split the cache for the same run
        raise ConfigError("data_parallel_load_balancing needs dp > 1")
    if lb == "hybrid" and fp > GPUS_PER_NODE:
        raise ConfigError("data_parallel_load_balancing hybrid: BLIS refuses a replica across nodes")
    return parallel, engine, cli, router


def engine_block(name, v, stack=V014):
    """A copy of the engine sub-block `name`, checked against SCENARIO_ENGINE_BLOCKS: a mapping of
    known fields, the required ones stated, each of its type (counts non-negative ints, not bools).
    A field `stack` measured inert (Stack.inert_block_fields) is refused with the reason."""
    inert = stack.inert_block_fields.get(name, {})
    fields = {f: t for f, t in SCENARIO_ENGINE_BLOCKS[name].items() if f not in inert}
    if not isinstance(v, dict):
        raise ConfigError(f"{name!r}: must be a mapping of {sorted(fields)}")
    for f in REQUIRED_BLOCK_FIELDS.get(name, ()):
        if f not in v:
            raise ConfigError(f"{name!r}: must state {f!r}")
    for f, x in v.items():
        if f in inert:
            raise ConfigError(f"{name!r}: {f!r}: {inert[f]}")
        if f not in fields:
            raise ConfigError(f"{name!r}: unknown field {f!r}; supported = {sorted(fields)}")
        t = fields[f]
        if type(x) is not t or (t is int and x < 0):
            want = "true or false" if t is bool else "a non-negative integer"
            raise ConfigError(f"{name!r}: {f!r} must be {want}, got {x!r}")
    return dict(v)


_EXPERTS_CACHE = {}


def model_experts(catalog, model):
    """The model's routed expert count as BLIS reads it: the largest `experts:` over the catalog
    graph's nodes (blis-vllm rules totalExperts), 0 for a graph with none (a dense model). A shared
    expert node (experts: 1) never wins over the routed one. Plain line scan: no yaml dependency.
    Raise ConfigError when the graph cannot be read."""
    key = (catalog, model)
    if key not in _EXPERTS_CACHE:
        path = os.path.join(catalog, "models", model, "graph.yaml")
        try:
            with open(path, encoding="utf-8") as f:
                counts = [int(line.split(":", 1)[1].split("#")[0]) for line in f
                          if line.strip().startswith("experts:")]
        except (OSError, ValueError) as e:
            raise ConfigError(f"cannot read the expert count from {path}: {e}") from None
        _EXPERTS_CACHE[key] = max(counts, default=0)
    return _EXPERTS_CACHE[key]


def check_eplb_layout(catalog, model, parallel, engine):
    """With EPLB on, refuse a layout whose EP width does not divide experts + redundant experts, and
    one that cannot be checked (no catalog, no model entry, no routed experts) rather than price it."""
    eplb = engine.get("eplb")
    if not eplb or not eplb.get("enable_eplb"):
        return
    if not catalog or not model:
        raise ConfigError("'eplb': cannot validate the EPLB layout without a catalog and a model")
    try:
        experts = model_experts(catalog, model)
    except ConfigError as e:
        raise ConfigError(f"'eplb': cannot validate the EPLB layout: {e}") from None
    if experts < 2:
        raise ConfigError(f"'eplb': cannot validate the EPLB layout: {model} has no routed experts")
    # the EP width, as blis-schemas v0.10.0 Parallelism.ExpertParallelWidth: TP x DP, not pp
    # (each pipeline stage holds its own layers' experts, sharded over that stage's TP x DP ranks)
    width = parallel["tp"] * parallel["dp"]
    redundant = eplb.get("num_redundant_experts", 0)
    if (experts + redundant) % width:
        need = (width - (experts + redundant) % width) % width
        raise ConfigError(f"'eplb': {experts} experts + {redundant} redundant = {experts + redundant}, "
                          f"which expert-parallel width {width} does not divide; {need} more redundant "
                          f"experts would make it divide")


def footprint_gpus(parallel):
    """GPUs one instance occupies = tp x pp x dp (EP is within the TP ranks, not an extra factor)."""
    return parallel["tp"] * parallel["pp"] * parallel["dp"]


_DSA_CACHE = {}


def is_dsa_model(catalog, model):
    """True when the model's catalog graph has a DSA sparse-MLA attention node.

    DSA is a property of the GRAPH, not of the model's name -- glm-5.3 and deepseek-v4-pro are DSA
    while qwen3-14b is not, and nothing in the names says so. Read it from the authoritative
    graph.yaml (NS-6: the catalog is the source of truth) with a plain substring scan so the runner
    keeps its no-external-yaml-dep property. An unreadable or absent graph answers False: the
    non-DSA default is the safe one, and BLIS reports the real error for a missing model anyway.
    """
    if not catalog or not model:
        return False
    key = (catalog, model)
    if key not in _DSA_CACHE:
        path = os.path.join(catalog, "models", model, "graph.yaml")
        try:
            with open(path, encoding="utf-8") as f:
                _DSA_CACHE[key] = "sparse_mla" in f.read()
        except OSError:
            _DSA_CACHE[key] = False
    return _DSA_CACHE[key]


def default_block_size(catalog, model):
    """The block_size to state when a candidate does not pin one.

    It cannot be a single constant in either direction: the kernel adapter REQUIRES a positive
    block_size on the deployment ("kernelmodel: deployment states no block_size"), so omitting it
    fails every model; but a flat 16 is refused by every DSA sparse-MLA model, which is what made
    glm-5 / glm-5.2 / glm-5.2-fp8 / glm-5.3 unrunnable in this runner.
    """
    return DSA_BLOCK_SIZE if is_dsa_model(catalog, model) else DEFAULT_BLOCK_SIZE


def _iter_json_objects(text):
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            if isinstance(obj, dict):
                yield obj
            i = end
        else:
            i += 1


def _find_metrics_dicts(obj, out):
    if isinstance(obj, dict):
        if "tokens_per_sec" in obj:
            out.append(obj)
        for v in obj.values():
            _find_metrics_dicts(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _find_metrics_dicts(v, out)
    return out


def _all_metrics(stdout):
    out = []
    for o in _iter_json_objects(stdout):
        _find_metrics_dicts(o, out)
    return out


def parse_metrics(stdout):
    cands = _all_metrics(stdout)
    for o in cands:
        if o.get("instance_id") == "cluster":
            return o
    return cands[0] if cands else {}


def parse_all_instances(stdout):
    return [o for o in _all_metrics(stdout) if o.get("instance_id") not in (None, "cluster")]


def _completion_issue(m):
    """Why a run is NOT a usable capacity/ranking point (incomplete/overloaded), else None."""
    inj, comp = m.get("injected_requests"), m.get("completed_requests")
    if inj is not None and comp is not None and inj != comp:
        return f"incomplete: injected={inj} != completed={comp}"
    for f in ("still_queued", "still_running", "dropped_unservable", "timed_out_requests"):
        v = m.get(f)
        if v:
            return f"overloaded: {f}={v}"
    return None


def pool_nodes(parallel, num_instances=1):
    """Nodes the colocated pool must hold for `num_instances` copies of this instance shape.

    BLIS (v0.10.0 on) validates `--num-instances` against the pool's rank capacity: a pool of
    `nodes` x 8 GPUs holds floor(nodes*8 / footprint) instances, and asking for more is refused
    ("N instance(s) requested, but ...'s colocated pool holds M rank(s) at its layout"). Sizing
    the pool from ONE instance's footprint therefore capped every search at a single node's
    worth of instances; size it for the whole fleet instead."""
    total = footprint_gpus(parallel) * max(1, num_instances)
    return max(1, -(-total // GPUS_PER_NODE))           # ceil(total / 8)


def scenario_yaml(model, hardware, engine_version, parallel, engine, name, num_instances=1,
                  fabric=DEFAULT_FABRIC, epp_file=None, coefficients=COEFFICIENTS):
    """Render a two-doc Scenario+Deployment YAML (no external yaml dep). Omitted engine keys (e.g.
    quantization when unset) let the checkpoint/engine govern. `epp_file` (a file name in the same
    directory) becomes the Deployment's router block; without it the router runs its built-in
    default configuration. `coefficients` are the stack's registry sets (Stack.coefficients)."""
    def block(d, indent):
        # a mapping value (an engine sub-block, e.g. eplb) is written as a nested block under its key
        pad = " " * indent
        return "\n".join(f"{pad}{k}:\n{block(v, indent + 2)}" if isinstance(v, dict)
                         else f"{pad}{k}: {json.dumps(v)}" for k, v in d.items())
    nodes = pool_nodes(parallel, num_instances)
    # A multi-node cluster MUST name a fabric: BLIS resolves cross-node cost from the fabric,
    # not the chip, and refuses the scenario otherwise ("scenario.cluster.fabric: required for
    # a N-node cluster"). Single-node clusters omit it (no cross-node traffic to price).
    fabric_line = f"\n  fabric: {fabric}" if nodes > 1 else ""
    return f"""kind: Scenario
name: {name}
engine_version: "{engine_version}"
model: {model}
coefficients: [{", ".join(coefficients)}]
cluster:
  hardware: {hardware}{fabric_line}
  nodes: {nodes}
  gpus_per_node: 8
---
kind: Deployment
name: {name}
pools:
  - role: colocated
    nodes: {nodes}
    parallel:
{block(parallel, 6)}
    engine:
{block(engine, 6)}
""" + (f"router:\n  endpoint_picker_config: {epp_file}\n" if epp_file else "")


def _publish(path, body):
    """Write `body` to `path` atomically (temp + os.replace) unless it exists: concurrent pool workers
    never read a half-written file, and a concurrent writer's replace is idempotent."""
    if os.path.exists(path):
        return
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(body)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_epp(scenarios_dir, epp_config):
    """Write an EndpointPickerConfig as JSON (valid YAML; no yaml dep), content-addressed; return its
    file name."""
    body = json.dumps(epp_config, sort_keys=True)
    name = f"epp-{hashlib.sha1(body.encode()).hexdigest()[:10]}.yaml"
    os.makedirs(scenarios_dir, exist_ok=True)
    _publish(os.path.join(scenarios_dir, name), body + "\n")
    return name


def write_scenario(scenarios_dir, model, hardware, engine_version, parallel, engine, num_instances=1,
                   fabric=DEFAULT_FABRIC, epp_file=None, coefficients=COEFFICIENTS):
    """Write a scenario ATOMICALLY (temp + os.replace) so concurrent pool workers never read a
    half-written file for the same hashed deployment; return its file name.

    `num_instances` participates in the hash because it sets the pool's node count — two runs
    that differ only in instance count need DIFFERENT scenario files, or the second reuses a
    pool too small for it and BLIS refuses the run. Coefficients other than v0.14.0's enter the hash
    too (so another stack never reuses a v0.14.0 file); v0.14.0's names keep the names they had."""
    key = [model, hardware, engine_version, parallel, engine, num_instances, fabric]
    if epp_file:
        key.append(epp_file)                # content-addressed itself, so the hash covers the config
    if list(coefficients) != COEFFICIENTS:
        key.append({"coefficients": list(coefficients)})
    key = json.dumps(key, sort_keys=True)
    name = f"{model}-{hardware}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"
    os.makedirs(scenarios_dir, exist_ok=True)
    _publish(os.path.join(scenarios_dir, name + ".yaml"),
             scenario_yaml(model, hardware, engine_version, parallel, engine, name, num_instances,
                           fabric, epp_file, coefficients))
    return name + ".yaml"


def resolve_defaults(blis, defaults):
    if defaults:
        return defaults
    beside = os.path.join(os.path.dirname(os.path.abspath(blis)), "defaults.yaml")
    return beside if os.path.exists(beside) else None


def build_cmd(blis, catalog, registry, scenarios_dir, scenario_name, shape, num_requests, seed,
              horizon, defaults, cli):
    # BLIS (v0.10.0 on) selects the kernel cost-model backend from the
    # scenario's `coefficients:` list (written into the generated scenario), not a CLI flag.
    # The old `--latency-model blis-latency-kernel` flag was removed in the release and is
    # rejected ("unknown flag"), so it must not be passed.
    cmd = [blis, "run",
           "--scenario", scenario_name, "--scenarios", scenarios_dir, "--registry", registry]
    if catalog:
        cmd += ["--catalog", catalog]
    if resolve_defaults(blis, defaults):
        cmd += ["--defaults-filepath", resolve_defaults(blis, defaults)]
    cmd += workloads.cli_flags(shape)       # not --workload: closed-loop runs ignore it
    if num_requests is not None:
        cmd += ["--num-requests", str(num_requests)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if horizon is not None:
        cmd += ["--horizon", str(horizon)]
    for k, v in cli.items():
        cmd += [CLI_FLAGS[k], str(v)]
    return cmd


def _record_shell(config, phase):
    return {"config": config, "concurrency": config.get("concurrency"),
            "config_hash": hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
            "phase": phase}


def run_one(blis, catalog, registry, scenarios_dir, model, hardware, engine_version,
            workload, num_requests, seed, horizon, config, phase="probe", timeout=900, defaults=None,
            stack=V014):
    """Run one kernel-priced BLIS config on `stack` and return a record dict. No runs.jsonl I/O
    (caller writes). Raises ConfigError for an engine_version not the stack's: a caller error, not
    a candidate's, so it is not recorded as an error row."""
    check_stack_engine(engine_version, stack)
    rec = _record_shell(config, phase)
    try:
        parallel, engine, cli, router = classify_config(config, engine_version, stack)
        check_eplb_layout(catalog, model, parallel, engine)
        shape = workloads.shape(workload or workloads.DISTRIBUTION, catalog)
    except (ConfigError, ValueError) as e:
        # Same row SHAPE as a real run so neither sweep_pool's summary nor the report trips on a
        # missing key (every record has wall_s, deployment, metrics, ...). returncode=None marks it
        # "not run"; error carries the reason.
        rec.update({"deployment": None, "cmd": None, "returncode": None, "wall_s": 0.0,
                    "metrics": {}, "metrics_all_instances": [], "error": f"unsupported config: {e}",
                    "stdout_tail": "", "stderr_tail": ""})
        return rec
    num_instances = cli.get("num_instances", 1)
    fp = footprint_gpus(parallel)
    # Fill block_size per MODEL when the candidate did not pin one (see default_block_size): the
    # adapter requires a value, and DSA models require it to be a multiple of 64.
    engine.setdefault("block_size", default_block_size(catalog, model))
    epp_file = write_epp(scenarios_dir, router["epp_config"]) if router else None
    scenario_name = write_scenario(scenarios_dir, model, hardware, engine_version, parallel, engine,
                                   num_instances, epp_file=epp_file, coefficients=stack.coefficients)
    cmd = build_cmd(blis, catalog, registry, scenarios_dir, scenario_name, shape,
                    num_requests, seed, horizon, defaults, cli)
    t0 = time.time()
    try:
        pr = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, rc = pr.stdout, pr.stderr, pr.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, rc = "", "timeout", -1
    metrics, per_instance, doc_error = {}, [], None
    if rc == 0 and stack.result_format == "result-doc":
        try:
            metrics, per_instance = result_doc.parse_result_document(stdout)
        except ValueError as e:
            doc_error = f"rc=0 but {e}"
    elif rc == 0:
        metrics, per_instance = parse_metrics(stdout), parse_all_instances(stdout)
    if rc != 0:
        error = stderr.strip().splitlines()[-1] if stderr.strip() else f"rc={rc}"
    elif doc_error:
        error = doc_error
    elif not metrics:
        error = "rc=0 but no cluster metrics parsed from stdout (output format changed?)"
    else:
        error = _completion_issue(metrics)
    rec.update({
        "deployment": {"parallel": parallel, "engine": engine, "num_instances": num_instances,
                       "footprint_gpus_per_instance": fp, "gpus": fp * num_instances,
                       "scenario": scenario_name, "epp_file": epp_file},
        "workload": {"name": workload or workloads.DISTRIBUTION, "shape": shape},
        "cmd": cmd, "returncode": rc, "wall_s": round(time.time() - t0, 3),
        "metrics": metrics, "metrics_all_instances": per_instance,
        "error": error, "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:],
    })
    return rec


def add_common_args(p):
    p.add_argument("--blis", default="./blis")
    p.add_argument("--catalog", default=os.environ.get("BLIS_CATALOG"))
    p.add_argument("--registry", default=os.environ.get("BLIS_REGISTRY"), help="blis-registry v0.2.0 clone root")
    p.add_argument("--scenarios", required=True, help="dir to write/read generated scenario YAMLs")
    p.add_argument("--model", required=True, help="catalog short-name, e.g. glm-5.3-flash")
    p.add_argument("--hardware", required=True, help="scenario hardware key, e.g. h100")
    p.add_argument("--engine-version", dest="engine_version", default="0.29.0")
    p.add_argument("--workload", help="catalog preset name or 'distribution' (default)")
    p.add_argument("--num-requests", type=int, dest="num_requests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int)
    p.add_argument("--defaults", help="path to defaults.yaml (default: next to the blis binary)")
    p.add_argument("--phase", default="probe")
    p.add_argument("--timeout", type=int, default=900)


def require_env(args):
    """Raise ValueError (do NOT sys.exit) -- this module is imported as a library; callers in a
    main() catch it and exit."""
    missing = []
    if not args.catalog:
        missing.append("--catalog or BLIS_CATALOG (NS-6)")
    if not args.registry:
        missing.append("--registry or BLIS_REGISTRY (blis-latency-kernel needs it)")
    if missing:
        raise ValueError("missing required: " + "; ".join(missing))


def main():
    p = argparse.ArgumentParser(description="Run one kernel-priced BLIS config; print its record")
    add_common_args(p)
    p.add_argument("--config", required=True, help='JSON knob dict, e.g. {"tp":8,"enable_expert_parallel":true,"num_instances":2,"concurrency":32}')
    p.add_argument("--out", help="optional runs.jsonl to append to (single-run use only)")
    args = p.parse_args()
    try:
        require_env(args)
    except ValueError as e:
        sys.exit(str(e))
    rec = run_one(args.blis, args.catalog, args.registry, args.scenarios, args.model, args.hardware,
                  args.engine_version, args.workload, args.num_requests, args.seed, args.horizon,
                  json.loads(args.config), args.phase, args.timeout, args.defaults)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
