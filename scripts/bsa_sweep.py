"""Sweep BSA generator settings on spaces whose grids are cached; report runs to within 1% per setting.

    uv run python scripts/bsa_sweep.py --cache cache/ --seeds 0-9 --budget 1000 \
        --case SPACE GRID [--case SPACE GRID ...] --settings '[{"alpha": 0.7}, {"alpha": 0.3, "batch": 32}]'

Every probe must be a cache hit from the grid: a cache miss raises CacheMiss and the sweep stops, so it
never launches BLIS. BLIS_CATALOG etc. must still be set: the cache key includes the workload's token
shape, which is read from the catalog. Each setting is a dict of BsaSearcher keyword arguments (alpha, rho, eps, batch, scheme) plus
"feas": true for the feasibility biaser. Output: one JSON line per (case, setting) with the median and
IQR of runs to within 1% (budget + 1 when never reached), the count of seeds that never reached it, and
the median regret at the budget (the best within `--budget` runs, not counting the last batch's
overshoot, so settings with different batch sizes compare at the same budget).

Each GRID must belong to its SPACE: its sidecar (GRID.meta.json, written by cf-grid) must exist and
match the space's space_hash and the current BLIS pins, or the sweep refuses to start, since the
optimum would be another space's."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config_forage.bench.analyze import load_grid, runs_to_within
from config_forage.bench.grid import meta_path
from config_forage.bench.run import parse_seeds
from config_forage.blis.runner import PINS, BlisEnv, CachedRunner, run_batch
from config_forage.feasibility import FeasibilityBiaser
from config_forage.probe import Objective
from config_forage.search.bsa import BsaSearcher
from config_forage.space import KnobSpace, load_space, space_hash


class CacheMiss(RuntimeError):
    """A probe the grid did not cache: the sweep would have to run BLIS."""


def _refuse(config, concurrency, seed):
    raise CacheMiss(f"cache miss at concurrency {concurrency}, seed {seed} for {config}: the cache does not "
                    "hold this space's grid for these seeds (run cf-grid first, or check --cache)")


def check_grid(space: KnobSpace, grid: Path) -> None:
    """Raise ValueError unless grid's sidecar exists and matches space's space_hash and the current pins."""
    mp = meta_path(grid)
    try:
        have = json.loads(mp.read_text())
    except FileNotFoundError:
        raise ValueError(f"{grid}: no sidecar {mp.name}, so its space and pins are unknown; "
                         "use a grid written by cf-grid") from None
    want = {"space_hash": space_hash(space), "pins": list(PINS)}
    bad = [k for k in want if have.get(k) != want[k]]
    if bad:
        raise ValueError(f"{grid} is not this space's grid at these pins: "
                         + "; ".join(f"{k} {have.get(k)} there, {want[k]} here" for k in bad))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs=2, action="append", metavar=("SPACE", "GRID"), required=True)
    ap.add_argument("--settings", required=True, help="JSON list of setting dicts")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    seeds, settings = parse_seeds(a.seeds), json.loads(a.settings)
    env = BlisEnv.from_env()
    spaces = [(load_space(s), Path(g), s) for s, g in a.case]
    for sp, grid, _ in spaces:  # every pairing is checked before any sweep starts
        check_grid(sp, grid)
    for sp, grid_path, space_path in spaces:
        opt = min(r["score"] for r in load_grid(grid_path) if r["score"] is not None)
        runner = CachedRunner(env, sp.scenario, a.cache, run_fn=_refuse)  # env: the shape enters the key
        for st in settings:
            kw = {k: v for k, v in st.items() if k != "feas"}
            biasers = [FeasibilityBiaser(sp)] if st.get("feas") else []

            def one(seed, kw=kw, biasers=biasers, sp=sp, runner=runner, opt=opt):
                t = BsaSearcher(sp, biasers, workers=1, **kw).search(Objective(sp, runner), a.budget, seed)
                r = runs_to_within(t, opt, a.budget)
                best = t.value_at(a.budget)
                return (a.budget + 1 if r is None else r), r is None, (None if best is None else best - opt)

            res = run_batch(one, seeds, a.jobs)
            runs = np.array([r for r, _, _ in res])
            regret = [g for _, _, g in res if g is not None]
            print(json.dumps({"case": Path(space_path).stem, "setting": st,
                              "runs_to_1pct_median": float(np.median(runs)),
                              "iqr": [float(np.percentile(runs, 25)), float(np.percentile(runs, 75))],
                              "censored": sum(c for _, c, _ in res), "seeds": len(seeds),
                              "regret_at_budget_median": float(np.median(regret)) if regret else None}),
                  flush=True)


if __name__ == "__main__":
    main()
