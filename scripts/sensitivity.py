"""One-axis sensitivity of tps_per_gpu around a base config, with real BLIS (cached).

    uv run python scripts/sensitivity.py SPACE --cache cache/ [--base '{"tp":4,"dp":2}'] [--workers 9]

Prints one JSON line per probed variant and a per-axis summary (spread = (max-min)/max over the
axis's scored values, in %). Infeasible topology variants are skipped."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config_forage.blis.runner import BlisEnv, CachedRunner, run_batch
from config_forage.feasibility import is_feasible
from config_forage.probe import probe
from config_forage.search.coord import _normalize
from config_forage.space import load_space


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("space", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--base", default="{}", help="JSON overrides of the default config")
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    sp = load_space(a.space)
    runner = CachedRunner(BlisEnv.from_env(), sp.scenario, a.cache)
    base = {k: v for k, v in sp.default_config().items() if k != "num_instances"}
    over = json.loads(a.base)
    unknown = sorted(set(over) - set(sp.names))
    if unknown:
        raise SystemExit(f"--base: not knobs of this space: {unknown}")
    base = _normalize(sp, {**base, **over})  # an override may switch a gated knob on or off
    dropped = sorted(k for k in over if k not in base)
    if dropped:
        raise SystemExit(f"--base: {dropped} do not apply (their `requires` is unmet at this base)")
    jobs = [("(base)", None)]
    gated_off = [k.name for k in sp.knobs if k.name not in base]
    for k in sp.knobs:
        if k.name in base:  # a knob gated off at the base (requires unmet) is not varied
            jobs += [(k.name, v) for v in k.values if v != base[k.name]]

    def one(job):
        name, v = job
        cfg = dict(base) if v is None else _normalize(sp, {**base, name: v})
        if not is_feasible(cfg, sp.scenario):
            return name, v, None
        return name, v, probe(runner, sp.config(sp.decision(cfg)), sp.scenario)

    t0 = time.monotonic()
    res = run_batch(one, jobs, a.workers)
    b = res[0][2]
    print(json.dumps({"base": base, "tps_per_gpu": b.tps_per_gpu, "c_star": b.c_star}))
    for name, v, p in res[1:]:
        if p is not None:
            print(json.dumps({"axis": name, "value": v, "tps_per_gpu": p.tps_per_gpu, "c_star": p.c_star,
                              "c_use": p.c_use, "diag": p.diagnostics}))
    print("| axis | value: tps/GPU (C*) | spread % |")
    print("|---|---|---|")
    for k in sp.knobs:
        if k.name not in base:
            continue
        pts = [(base[k.name], b)] + [(v, p) for n, v, p in res[1:] if n == k.name and p is not None]
        pts.sort(key=lambda x: k.values.index(x[0]))
        vals = [p.tps_per_gpu for _, p in pts if p.tps_per_gpu is not None]
        spread = 100 * (max(vals) - min(vals)) / max(vals) if len(vals) > 1 else 0.0
        cells = ", ".join(f"{v}{'*' if v == base[k.name] else ''}: "
                          + ("fail" if p.tps_per_gpu is None else f"{p.tps_per_gpu:.1f} ({p.c_star})")
                          for v, p in pts)
        flag = " **<1%**" if spread < 1 else ""
        print(f"| {k.name} | {cells} | {spread:.1f}{flag} |")
    if gated_off:
        print(f"gated off at the base (not varied): {gated_off}")
    print(f"unique BLIS runs {runner.unique_runs}, cache hits {runner.cache_hits}, "
          f"wall {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
