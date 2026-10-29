"""Measure real BLIS cost per config on a random sample of a space's feasible decisions.

    uv run python scripts/grid_cost.py SPACE --cache DIR --n 300 --workers 9 [--seed 0]

Use a fresh cache dir so every run is a real BLIS call; prints unique runs/config, wall/config and
the extrapolation to the full grid."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from config_forage.blis.runner import BlisEnv, CachedRunner, run_batch
from config_forage.feasibility import enumerate_feasible
from config_forage.probe import probe
from config_forage.space import load_space


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("space", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sp = load_space(a.space)
    feas = list(enumerate_feasible(sp))
    idx = np.random.default_rng(a.seed).choice(len(feas), size=min(a.n, len(feas)), replace=False)
    sample = [feas[i] for i in idx]
    runner = CachedRunner(BlisEnv.from_env(), sp.scenario, a.cache)
    walls: list[float] = []
    run0 = runner.run_fn

    def timed(config, c, seed):
        rec = run0(config, c, seed)
        walls.append(rec.get("wall_s") or 0.0)
        return rec

    runner.run_fn = timed
    t0 = time.monotonic()
    res = run_batch(lambda d: probe(runner, sp.config(d), sp.scenario), sample, a.workers)
    wall = time.monotonic() - t0
    scored = [p for p in res if p.score is not None]
    n = len(sample)
    print(f"feasible {len(feas)}; sampled {n}; scored {len(scored)}")
    print(f"unique BLIS runs {runner.unique_runs} ({runner.unique_runs / n:.2f}/config; "
          f"{statistics.mean(p.runs for p in scored) if scored else 0:.2f}/scored config)")
    print(f"BLIS wall_s per run: median {statistics.median(walls):.3f}, mean {statistics.mean(walls):.3f}")
    print(f"elapsed {wall:.1f}s at --workers {a.workers} = {wall / n:.3f}s/config")
    print(f"full-grid estimate: {len(feas) * runner.unique_runs / n:.0f} runs, "
          f"{len(feas) * wall / n / 60:.1f} min at --workers {a.workers}")


if __name__ == "__main__":
    main()
