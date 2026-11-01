"""Sweep BSA generator settings on spaces whose grids are cached; report runs to within 1% per setting.

    uv run python scripts/bsa_sweep.py --cache cache/ --seeds 0-9 --budget 1000 \
        --case SPACE GRID [--case SPACE GRID ...] --settings '[{"alpha": 0.7}, {"alpha": 0.3, "batch": 32}]'

Every probe must be a cache hit from the grid, so a sweep makes no new BLIS runs (it reports any it
made). Each setting is a dict of BsaSearcher keyword arguments (alpha, rho, eps, batch, scheme) plus
"feas": true for the feasibility biaser. Output: one JSON line per (case, setting) with the median and
IQR of runs to within 1% (budget + 1 when never reached), the count of seeds that never reached it, and
the median final regret."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config_forage.bench.analyze import load_grid, runs_to_within
from config_forage.bench.run import parse_seeds
from config_forage.blis.runner import BlisEnv, CachedRunner, run_batch
from config_forage.feasibility import FeasibilityBiaser
from config_forage.probe import Objective
from config_forage.search.bsa import BsaSearcher
from config_forage.space import load_space


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
    for space_path, grid_path in a.case:
        sp = load_space(space_path)
        opt = min(r["score"] for r in load_grid(Path(grid_path)) if r["score"] is not None)
        runner = CachedRunner(env, sp.scenario, a.cache)
        for st in settings:
            kw = {k: v for k, v in st.items() if k != "feas"}
            biasers = [FeasibilityBiaser(sp)] if st.get("feas") else []

            def one(seed, kw=kw, biasers=biasers):
                t = BsaSearcher(sp, biasers, workers=1, **kw).search(Objective(sp, runner), a.budget, seed)
                r = runs_to_within(t, opt, a.budget)
                best = t.points[-1].best_score if t.points else None
                return (a.budget + 1 if r is None else r), r is None, (None if best is None else best - opt)

            before = runner.unique_runs
            res = run_batch(one, seeds, a.jobs)
            runs = np.array([r for r, _, _ in res])
            regret = [g for _, _, g in res if g is not None]
            print(json.dumps({"case": Path(space_path).stem, "setting": st,
                              "runs_to_1pct_median": float(np.median(runs)),
                              "iqr": [float(np.percentile(runs, 25)), float(np.percentile(runs, 75))],
                              "censored": sum(c for _, c, _ in res), "seeds": len(seeds),
                              "final_regret_median": float(np.median(regret)) if regret else None,
                              "new_blis_runs": runner.unique_runs - before}), flush=True)


if __name__ == "__main__":
    main()
