"""Workload shapes: a workload name resolved to the token distribution BLIS is told to draw from.

BLIS ignores `--workload` in concurrency mode: `blis run --concurrency` synthesizes requests from
the `--prompt-tokens*` / `--output-tokens*` / `--prefix-tokens` flags, whose defaults are a 512/512
"distribution" workload, and reads a named preset only in rate mode (inference-sim v0.14.0
cmd/root.go:946-969; v0.10.0 has the same branch order). The probe always runs closed-loop, so a
named preset must be passed as those flags. A shape is that flag set; it also enters the cache key,
so a run is cached under the distribution it ran, not under the name it was given.

`distribution` is BLIS's flag defaults (cmd/root.go:39-46). Any other name is read from
<catalog>/workloads/<name>.yaml with the keys BLIS's preset reader accepts; a replay workload
(top-level `replay:`) is refused, as it has no shape."""

from __future__ import annotations

from pathlib import Path

import yaml

DISTRIBUTION = "distribution"
# BLIS v0.14.0 flag defaults (cmd/root.go:39-46): what every concurrency-mode run drew from before
DISTRIBUTION_SHAPE = {"prefix_tokens": 0,
                      "prompt": {"tokens": 512, "tokens_stdev": 256, "tokens_min": 2, "tokens_max": 7000},
                      "output": {"tokens": 512, "tokens_stdev": 256, "tokens_min": 2, "tokens_max": 7000}}
_SIDE_KEYS = ("tokens", "tokens_stdev", "tokens_min", "tokens_max")


def shape(name: str, catalog: str | Path | None) -> dict:
    """The shape of workload `name`: DISTRIBUTION_SHAPE, or the catalog preset's. Raises ValueError
    for a replay workload, a missing preset, or a preset with unknown or missing keys."""
    if name == DISTRIBUTION:
        return DISTRIBUTION_SHAPE
    if catalog is None:
        raise ValueError(f"workload {name!r}: needs a catalog to resolve (or use {DISTRIBUTION!r})")
    path = Path(catalog) / "workloads" / f"{name}.yaml"
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raise ValueError(f"workload {name!r}: no preset at {path}") from None
    raw = {k: v for k, v in raw.items() if not k.startswith("_comment")}
    if "replay" in raw:
        raise ValueError(f"workload {name!r}: a replay workload has no token shape; replay is not supported")
    unknown = set(raw) - {"prefix_tokens", "prompt", "output"}
    if unknown:
        raise ValueError(f"workload {name!r}: unknown keys {sorted(unknown)} in {path}")
    out = {"prefix_tokens": raw.get("prefix_tokens", 0)}
    for side in ("prompt", "output"):
        d = raw.get(side) or {}
        if set(d) != set(_SIDE_KEYS):
            raise ValueError(f"workload {name!r}: {side} must state exactly {list(_SIDE_KEYS)} ({path})")
        out[side] = dict(d)
    problem = _bounds_problem(out)
    if problem:
        raise ValueError(f"workload {name!r}: {problem} ({path})")
    return out


def _bounds_problem(s: dict) -> str | None:
    """Why BLIS would refuse shape `s`, else None: the checks of its validateDistributionParams
    (inference-sim v0.14.0 cmd/root.go:199-231) and --prefix-tokens >= 0 (cmd/root.go:904), so a
    malformed preset is refused once, here, not by a failed BLIS run per candidate."""
    for side in ("prompt", "output"):
        for k in _SIDE_KEYS:
            v = s[side][k]
            if not isinstance(v, int) or isinstance(v, bool):
                return f"{side}.{k} must be an integer, got {v!r}"
    p = s["prefix_tokens"]
    if not isinstance(p, int) or isinstance(p, bool) or p < 0:
        return f"prefix_tokens must be an integer >= 0, got {p!r}"
    for side in ("prompt", "output"):
        mean, sd, lo, hi = (s[side][k] for k in _SIDE_KEYS)
        if lo < 1 or hi < 1:
            return f"{side}.tokens_min and tokens_max must be >= 1, got {lo} and {hi}"
        if sd < 0:
            return f"{side}.tokens_stdev must be >= 0, got {sd}"
        if lo > hi:
            return f"{side}.tokens_min ({lo}) must be <= tokens_max ({hi})"
        if not lo <= mean <= hi or sd > hi or (sd != 0 and sd < lo):
            return f"{side}.tokens and tokens_stdev must be in [tokens_min, tokens_max] = [{lo}, {hi}]"
    return None


def cli_flags(s: dict) -> list[str]:
    """The `blis run` flags that make a closed-loop run draw from shape `s`."""
    flags = ["--prefix-tokens", str(s["prefix_tokens"])]
    for side, stem in (("prompt", "--prompt-tokens"), ("output", "--output-tokens")):
        d = s[side]
        flags += [stem, str(d["tokens"]), f"{stem}-stdev", str(d["tokens_stdev"]),
                  f"{stem}-min", str(d["tokens_min"]), f"{stem}-max", str(d["tokens_max"])]
    return flags
