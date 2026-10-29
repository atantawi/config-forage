"""Run IncumbentSearcher and CoordAscentSearcher on a space with real (cached) BLIS; compare to a grid.

    uv run python scripts/baselines_vs_grid.py SPACE --grid grid.jsonl --cache cache/ [--budget 1000]

Each searcher gets a fresh Objective (runs_used counts distinct cache keys, hits included), so runs
are what the method would pay on a cold cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from config_forage.bench.analyze import load_grid
from config_forage.blis.runner import BlisEnv, CachedRunner
from config_forage.probe import Objective
from config_forage.search.coord import CoordAscentSearcher, IncumbentSearcher
from config_forage.space import load_space


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("space", type=Path)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    sp = load_space(a.space)
    rows = [r for r in load_grid(a.grid) if r["score"] is not None]
    opt = min(r["score"] for r in rows)
    plateau = [r for r in rows if r["score"] <= opt * 0.99]  # scores are negative
    print(f"grid: {len(rows)} scored; optimum {opt:.3f}; within 1%: {len(plateau)}")
    runner = CachedRunner(BlisEnv.from_env(), sp.scenario, a.cache)
    runs = [("incumbent", IncumbentSearcher(sp), 0)]
    runs += [("coord-ascent", CoordAscentSearcher(sp), s) for s in range(a.seeds)]
    for name, searcher, seed in runs:
        obj = Objective(sp, runner)
        t = searcher.search(obj, a.budget, seed)
        best = t.points[-1].best_score if t.points else None
        gap = None if best is None else 100 * (best - opt) / abs(opt)
        cfg = t.points[-1].best_config if t.points else None
        within = next((p.runs for p in t.points if p.best_score is not None and p.best_score <= opt * 0.99), None)
        print(f"{name} seed {seed}: best {best:.3f} gap {gap:.2f}% runs {obj.runs_used} evals {obj.evaluations} "
              f"stop {t.extra.get('stop')} runs-to-within-1% {within}")
        print(f"   {cfg}")


if __name__ == "__main__":
    main()
