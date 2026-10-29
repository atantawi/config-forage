"""Knob space: typed serving knobs loaded from YAML, mapped onto a BSA DecisionSpace.

Adding a knob is a pure YAML addition; `tp`, `dp`, `enable_expert_parallel` also drive feasibility."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from config_forage.bsa.biaser import ApplicabilityBiaser
from config_forage.bsa.space import Axis, Decision, DecisionSpace


@dataclass(frozen=True)
class Slo:
    itl_p99_ms: float = 20.0
    ttft_p99_ms: float | None = None

    def passes(self, metrics: dict) -> bool:
        """All required p99 metrics are present and within bound; a missing metric fails."""
        for key, bound in (("itl_p99_ms", self.itl_p99_ms), ("ttft_p99_ms", self.ttft_p99_ms)):
            if bound is not None and not (metrics.get(key) is not None and metrics[key] <= bound):
                return False
        return True


@dataclass(frozen=True)
class Scenario:
    model: str
    hardware: str
    workload: str
    gpu_budget: int
    engine_version: str
    slo: Slo = field(default_factory=Slo)
    headroom: float = 0.8
    seed: int = 0
    num_requests_base: int = 80
    model_entry: dict = field(default_factory=dict)  # arch (MoE|dense), heads_kv, min_footprint (tp·dp floor); absent = dense


@dataclass(frozen=True)
class Knob:
    name: str
    values: tuple
    requires: tuple[str, tuple] | None = None  # applies only when knob requires[0] takes one of requires[1]
    default: object = None

    def __post_init__(self):
        if not self.values:
            raise ValueError(f"knob {self.name}: no values")
        if self.default is None:
            object.__setattr__(self, "default", self.values[0])
        elif self.default not in self.values:
            raise ValueError(f"knob {self.name}: default {self.default!r} not in {self.values}")


@dataclass(frozen=True)
class KnobSpace:
    scenario: Scenario
    knobs: tuple[Knob, ...]

    def __post_init__(self):
        names = [k.name for k in self.knobs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate knob names in {names}")
        if "num_instances" in names:
            raise ValueError("num_instances is derived, not a knob")
        topology = ("tp", "dp", "enable_expert_parallel")  # = feasibility.TOPOLOGY (no import cycle)
        for i, k in enumerate(self.knobs):
            labels = [str(v) for v in k.values]
            if len(set(labels)) != len(labels):
                raise ValueError(f"knob {k.name}: option labels {labels} are not unique (str of each value)")
            if k.requires is not None and k.name in topology:
                raise ValueError(f"knob {k.name}: a topology knob cannot be conditional (requires)")
            if k.requires is not None:
                if k.requires[0] not in names[:i]:
                    raise ValueError(f"knob {k.name}: requires an earlier knob, not {k.requires[0]!r}")
                dep = self.knobs[names.index(k.requires[0])]
                if not all(v in dep.values for v in k.requires[1]):
                    raise ValueError(f"knob {k.name}: requires values {k.requires[1]} not in {dep.name}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(k.name for k in self.knobs)

    @property
    def decision_space(self) -> DecisionSpace:
        return DecisionSpace(tuple(Axis(k.name, "", tuple(str(v) for v in k.values)) for k in self.knobs))

    def n_combinations(self) -> int:
        return math.prod(len(k.values) for k in self.knobs)

    def applicability_biaser(self) -> ApplicabilityBiaser | None:
        """The structural biaser for the knobs' `requires` (None when no knob has one): an axis is
        left unlabelled where its parent knob does not take a required value. Applicability is
        structure, not a bias, so BsaSearcher and RandomSearcher always apply it first."""
        names = self.names
        req = {m: (names.index(k.requires[0]),
                   tuple(self.knobs[names.index(k.requires[0])].values.index(v) for v in k.requires[1]))
               for m, k in enumerate(self.knobs) if k.requires}
        return ApplicabilityBiaser(self.decision_space, req) if req else None

    def canonical(self, d: Decision) -> Decision:
        """d with every knob whose `requires` is unmet set to None (unlabelled), in knob order, so a
        parent that is itself unlabelled also unlabels its dependants. Idempotent."""
        names, out = self.names, list(d)
        for i, k in enumerate(self.knobs):
            if k.requires is not None and out[i] is not None:
                p = names.index(k.requires[0])
                if out[p] is None or self.knobs[p].values[out[p]] not in k.requires[1]:
                    out[i] = None
        return tuple(out)

    def knob_values(self, d: Decision) -> dict:
        """The decision's labelled knob values, typed (no num_instances); defined for any decision."""
        return {k.name: k.values[i] for k, i in zip(self.knobs, d) if i is not None}

    def config(self, d: Decision) -> dict:
        """Typed config of a decision; unlabelled axes are omitted, num_instances is derived.
        Raises ValueError when the instance width tp·dp exceeds the GPU budget (num_instances 0):
        check `is_feasible(knob_values(d), scenario)` first for a decision that may be infeasible."""
        c = self.knob_values(d)
        width = c.get("tp", 1) * c.get("dp", 1)
        c["num_instances"] = self.scenario.gpu_budget // width
        if c["num_instances"] == 0:
            raise ValueError(f"num_instances 0: width tp·dp = {width} exceeds the GPU budget "
                             f"{self.scenario.gpu_budget}")
        return c

    def decision(self, config: dict) -> Decision:
        return tuple(k.values.index(config[k.name]) if k.name in config else None for k in self.knobs)

    def default_config(self) -> dict:
        """Each knob at its default (conditional knobs off where inapplicable); the topology must be feasible."""
        from config_forage.feasibility import is_feasible
        c: dict = {}
        for k in self.knobs:
            if k.requires is None or c.get(k.requires[0]) in k.requires[1]:
                c[k.name] = k.default
        if not is_feasible(c, self.scenario):
            topo = {n: c[n] for n in ("tp", "dp", "enable_expert_parallel") if n in c}
            raise ValueError(f"default topology {topo} is infeasible for the scenario")
        return self.config(self.decision(c))


def space_hash(space: KnobSpace) -> str:
    """sha256 of the canonical JSON of the parsed scenario and knobs (names, values, requires,
    defaults, order): what a space means, not its file bytes, so comments and whitespace do not
    change it. cf-bench and cf-grid store it and refuse to resume into results of another space."""
    payload = {"scenario": asdict(space.scenario), "knobs": [asdict(k) for k in space.knobs]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def load_space(path: str | Path) -> KnobSpace:
    raw = yaml.safe_load(Path(path).read_text())
    s = dict(raw["scenario"])
    s["slo"] = Slo(**s.get("slo", {}))
    knobs = tuple(Knob(k["name"], tuple(k["values"]),
                       (k["requires"][0], tuple(k["requires"][1])) if k.get("requires") else None,
                       k.get("default")) for k in raw["knobs"])
    return KnobSpace(Scenario(**s), knobs)
