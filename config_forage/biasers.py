"""Search biasers built on probe feedback: diagnostic rules and a transfer prior.

Both are soft (factors finite and > 0) and return ones, never None: applicability is left to the
structural biasers, which should come first in the generator's biaser list. Rules address knobs by
name, so adding a knob to a space YAML is a pure addition; a rule on an absent knob is ignored."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config_forage.bsa.space import Decision
from config_forage.probe import ProbeResult
from config_forage.space import KnobSpace

SIGNALS = ("kv_starved", "ttft_bound", "itl")  # itl: diagnostics["binding"] == "itl"
MAX_LOG = 50.0  # cap on |log factor| per axis, so exp never overflows whatever beta is


@dataclass(frozen=True)
class Rule:
    """When elites show `signal`, favour `knob` values: "higher", "lower", or a concrete value."""
    signal: str
    knob: str
    prefer: object


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("kv_starved", "gpu_memory_utilization", "higher"),
    Rule("kv_starved", "cache_dtype", "fp8"),
    Rule("kv_starved", "max_num_seqs", "lower"),
    Rule("ttft_bound", "max_num_batched_tokens", "higher"),
    Rule("itl", "max_num_seqs", "lower"),
)


def _shows(r: ProbeResult, signal: str) -> bool:
    d = r.diagnostics or {}
    return d.get("binding") == "itl" if signal == "itl" else bool(d.get(signal))


class DiagnosticBiaser:
    """factor = exp(beta * sum over matching rules of f_s * z), z per rule:

    - "higher"/"lower": z runs linearly from -1 to +1 over the value index (knob values are
      assumed listed in ascending order, as in our spaces), flipped for "lower"; a single-value
      knob gets z = 0;
    - a concrete value: z = 1 on that value, 0 elsewhere (rule ignored if the value is absent).

    f_s is the fraction of the top-rho elites (successes only, at least 1) of the last observed
    generation that show signal s. All ones before the first observe; a generation with no
    success keeps the previous factors."""

    def __init__(self, space: KnobSpace, rules: Sequence[Rule], beta: float = 1.0, rho: float = 0.3):
        bad = [r.signal for r in rules if r.signal not in SIGNALS]
        if bad:
            raise ValueError(f"unknown signal(s) {bad}; expected one of {SIGNALS}")
        self.space, self.beta, self.rho = space, beta, rho
        names = space.names
        self.rules: list[tuple[str, int, np.ndarray]] = []  # (signal, axis, z)
        self._rule_of: list[Rule] = []  # the Rule behind each self.rules entry
        for r in rules:
            if r.knob not in names:
                continue
            m = names.index(r.knob)
            values = space.knobs[m].values
            n = len(values)
            if r.prefer in ("higher", "lower"):
                z = np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(1)
                if r.prefer == "lower":
                    z = -z
            elif r.prefer in values:
                z = np.zeros(n)
                z[values.index(r.prefer)] = 1.0
            else:
                continue
            self.rules.append((r.signal, m, z))
            self._rule_of.append(r)
        self._factors = [np.ones(len(k.values)) for k in space.knobs]
        self.fractions: dict[str, float] = {s: 0.0 for s in SIGNALS}

    def active_rules(self) -> list[Rule]:
        """The rules that can ever move a factor in this space: the knob has more than one value, z
        is not all zero, and the signal can occur (ttft_bound only under a TTFT SLO). An empty list
        means this biaser is inert: its factors stay all ones."""
        ttft = self.space.scenario.slo.ttft_p99_ms is not None
        return [r for r, (s, m, z) in zip(self._rule_of, self.rules)
                if len(self.space.knobs[m].values) > 1 and np.any(z) and (s != "ttft_bound" or ttft)]

    def observe(self, results: Sequence[tuple[Decision, ProbeResult]]) -> None:
        ok = sorted((r for _, r in results if r.score is not None), key=lambda r: r.score)
        if not ok:
            return
        elites = ok[:max(1, round(self.rho * len(ok)))]
        self.fractions = {s: sum(_shows(r, s) for r in elites) / len(elites) for s in SIGNALS}
        logs = [np.zeros(len(k.values)) for k in self.space.knobs]
        for s, m, z in self.rules:
            logs[m] += self.fractions[s] * z
        self._factors = [np.exp(np.clip(self.beta * lg, -MAX_LOG, MAX_LOG)) for lg in logs]

    def factor(self, axis: int, partial: Decision) -> np.ndarray:
        return self._factors[axis].copy()


class TransferPrior:
    """factor(axis) = snapshot[axis] ** beta: a P snapshot from an earlier search, tempered.
    beta = 0 is neutral, beta = 1 the snapshot itself. Entries must be finite and > 0 (BSA's
    eps floor guarantees it for a generator snapshot)."""

    def __init__(self, snapshot: Sequence[Sequence[float]], beta: float = 0.5, space: KnobSpace | None = None):
        snap = [np.asarray(p, dtype=float) for p in snapshot]
        if any(p.ndim != 1 or not np.all(np.isfinite(p)) or not np.all(p > 0) for p in snap):
            raise ValueError("snapshot entries must be finite and positive")
        if space is not None:
            want, got = tuple(space.decision_space.sizes), tuple(len(p) for p in snap)
            if want != got:
                raise ValueError(f"snapshot axis sizes {got} do not match the space's sizes {want}")
        self.beta = beta
        self._factors = [np.exp(np.clip(beta * np.log(p), -MAX_LOG, MAX_LOG)) for p in snap]

    @classmethod
    def from_history(cls, path: str | Path, beta: float = 0.5, space: KnobSpace | None = None) -> TransferPrior:
        """The last generation's P from a BSA trace JSON (Trace.extra["history"])."""
        hist = json.loads(Path(path).read_text()).get("extra", {}).get("history")
        if not hist:
            raise ValueError(f"{path}: no BSA history in the trace")
        return cls(hist[-1]["P"], beta, space)

    def factor(self, axis: int, partial: Decision) -> np.ndarray:
        return self._factors[axis].copy()
