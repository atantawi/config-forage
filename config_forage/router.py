"""llm-d router knobs -> an EndpointPickerConfig.

BLIS v0.12.0 on routes, admits and flow-controls only through the embedded llm-d router, configured
by the EndpointPickerConfig the scenario's `router:` block names. A space states the router with flat
knobs: `endpoint_picker_config` picks a family (the configs BLIS v0.14.0 ships in examples/llm-d), and
`<scorer>_weight` knobs override that family's scorer weights. `to_blis` turns them into the config
dict run_blis writes next to the scenario. Without `endpoint_picker_config` the router runs its
built-in default (queue 2, kv-cache-utilization 2, prefix-cache 3), and no weight knob may be set."""

from __future__ import annotations

# family -> (producer plugins, {scorer: default weight}), as in inference-sim v0.14.0 examples/llm-d
FAMILIES: dict[str, tuple[tuple[str, ...], dict[str, float]]] = {
    "load-only": ((), {"queue-scorer": 1, "kv-cache-utilization-scorer": 1}),
    "inflight-load": (("inflight-load-producer",),
                      {"active-request-scorer": 2, "queue-scorer": 1, "kv-cache-utilization-scorer": 1,
                       "prefix-cache-scorer": 1}),
    "prefix-affinity": (("approx-prefix-cache-producer",),
                        {"prefix-cache-scorer": 3, "queue-scorer": 1, "kv-cache-utilization-scorer": 1}),
}
FAMILY_KNOB = "endpoint_picker_config"
WEIGHT_KNOBS = {"queue_scorer_weight": "queue-scorer",
                "kv_cache_utilization_scorer_weight": "kv-cache-utilization-scorer",
                "prefix_cache_scorer_weight": "prefix-cache-scorer",
                "active_request_scorer_weight": "active-request-scorer"}
ROUTER_KNOBS = (FAMILY_KNOB, *WEIGHT_KNOBS)


def endpoint_picker_config(family: str, weights: dict[str, float] | None = None) -> dict:
    """The EndpointPickerConfig of `family` with scorer weights overridden by `weights` ({scorer: w})."""
    if family not in FAMILIES:
        raise ValueError(f"{FAMILY_KNOB} {family!r}: not one of {sorted(FAMILIES)}")
    producers, scorers = FAMILIES[family]
    weights = weights or {}
    stray = sorted(set(weights) - set(scorers))
    if stray:
        raise ValueError(f"{FAMILY_KNOB} {family!r} has no scorer {stray}")
    picked = {s: weights.get(s, w) for s, w in scorers.items()}
    picker = "max-score-picker"
    return {"apiVersion": "llm-d.ai/v1", "kind": "EndpointPickerConfig",
            "plugins": [{"type": t} for t in ("single-profile-handler", *producers, *scorers, picker)],
            "schedulingProfiles": [{"name": "default",
                                    "plugins": [*({"pluginRef": s, "weight": w} for s, w in picked.items()),
                                                {"pluginRef": picker}]}]}


def to_blis(config: dict) -> dict:
    """config with its router knobs replaced by `epp_config` (an EndpointPickerConfig); identity when
    it has none. Raises ValueError for a weight knob without a family, or one the family lacks."""
    if not any(k in config for k in ROUTER_KNOBS):
        return config
    rest = {k: v for k, v in config.items() if k not in ROUTER_KNOBS}
    weights = {WEIGHT_KNOBS[k]: config[k] for k in WEIGHT_KNOBS if k in config}
    if FAMILY_KNOB not in config:
        raise ValueError(f"router weight knobs {sorted(k for k in WEIGHT_KNOBS if k in config)} "
                         f"need {FAMILY_KNOB}")
    return {**rest, "epp_config": endpoint_picker_config(config[FAMILY_KNOB], weights)}
