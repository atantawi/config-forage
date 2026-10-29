"""cf-bench: run each arm at each seed to a budget of unique BLIS runs over one shared cached runner.

Layout: OUT/<arm>/seed-<k>.json (Trace JSON) and OUT/manifest.json; existing traces are skipped."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config_forage.biasers import DEFAULT_RULES, DiagnosticBiaser, TransferPrior
from config_forage.blis.runner import PINS, BlisEnv, CachedRunner
from config_forage.feasibility import FeasibilityBiaser
from config_forage.probe import Objective
from config_forage.search.base import Searcher, Trace
from config_forage.search.bsa import BsaSearcher
from config_forage.search.coord import CoordAscentSearcher, IncumbentSearcher
from config_forage.search.random import RandomSearcher
from config_forage.space import KnobSpace, load_space


def _transfer(sp: KnobSpace, o: dict) -> Searcher:
    if not o.get("transfer_from"):
        raise ValueError("arm bsa-transfer needs the transfer_from option (--transfer-from TRACE.json)")
    return BsaSearcher(sp, [FeasibilityBiaser(sp), TransferPrior.from_history(o["transfer_from"], space=sp)],
                       workers=o.get("workers"), name="bsa-transfer")


def _tpe(sp: KnobSpace, o: dict) -> Searcher:
    from config_forage.search.tpe import (
        TpeSearcher,  # optuna import is slow; load on use
    )
    return TpeSearcher(sp)


def _smac(sp: KnobSpace, o: dict) -> Searcher:
    from config_forage.search.smac import SmacSearcher
    return SmacSearcher(sp)


# Each call builds a fresh searcher (and fresh biasers: they carry state).
ARMS: dict[str, Callable[[KnobSpace, dict], Searcher]] = {
    "bsa": lambda sp, o: BsaSearcher(sp, [], workers=o.get("workers"), name="bsa"),
    "bsa-feas": lambda sp, o: BsaSearcher(sp, [FeasibilityBiaser(sp)], workers=o.get("workers"), name="bsa-feas"),
    "bsa-diag": lambda sp, o: BsaSearcher(sp, [FeasibilityBiaser(sp), DiagnosticBiaser(sp, DEFAULT_RULES)],
                                          workers=o.get("workers"), name="bsa-diag"),
    "bsa-transfer": _transfer,
    "random": lambda sp, o: RandomSearcher(sp, feasible_only=True, workers=o.get("workers")),
    "tpe": _tpe,
    "smac": _smac,
    "incumbent": lambda sp, o: IncumbentSearcher(sp),
    "coord-ascent": lambda sp, o: CoordAscentSearcher(sp),
}


def parse_seeds(s: str) -> list[int]:
    """'0-9' or '0,3,5-7' -> sorted distinct ints."""
    out: set[int] = set()
    for part in s.split(","):
        lo, _, hi = part.strip().partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def run_one(space: KnobSpace, runner, arm: str, seed: int, budget: int, opts: dict) -> Trace:
    """A fresh searcher and a fresh Objective over the shared runner; stop reason filled if absent."""
    t = ARMS[arm](space, opts).search(Objective(space, runner), budget, seed)
    t.method = arm
    if "stop" not in t.extra:
        t.extra["stop"] = "budget" if t.points and t.points[-1].runs >= budget else "stall"
    return t


def bench(space: KnobSpace, runner, arms: list[str], seeds: list[int], budget: int, out: Path,
          opts: dict | None = None, jobs: int = 1, space_path: str | None = None) -> list[Trace]:
    """Run every (arm, seed) whose trace file is missing; return the new traces in (arm, seed) order."""
    opts, out = dict(opts or {}), Path(out)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; expected some of {list(ARMS)}")
    if "bsa-transfer" in arms and not opts.get("transfer_from"):
        raise ValueError("arm bsa-transfer needs the transfer_from option (--transfer-from TRACE.json)")
    if jobs > 1 and opts.get("workers") is None:  # keep jobs x workers threads near the core count
        opts["workers"] = max(1, ((os.cpu_count() or 2) - 1) // jobs)
    mpath, old = out / "manifest.json", {}
    if mpath.exists():
        old = json.loads(mpath.read_text())
        if old.get("budget") != budget:
            raise ValueError(f"{out} holds runs at budget {old.get('budget')}, not {budget}")
    out.mkdir(parents=True, exist_ok=True)
    all_arms = list(dict.fromkeys([*old.get("arms", []), *arms]))  # a resumed call adds to the record
    all_seeds = sorted({*old.get("seeds", []), *seeds})
    mpath.write_text(json.dumps({"space": space_path, "budget": budget, "arms": all_arms, "seeds": all_seeds,
                                 "pins": list(PINS), "git_sha": _git_sha(), "jobs": jobs,
                                 "workers": opts.get("workers"),
                                 "opts": {k: str(v) for k, v in opts.items() if k != "workers"}}, indent=2))
    todo = [(a, s) for a in arms for s in seeds if not (out / a / f"seed-{s}.json").exists()]

    def one(job):
        a, s = job
        t = run_one(space, runner, a, s, budget, opts)
        p = out / a / f"seed-{s}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(t.to_json())
        tmp.replace(p)
        return t

    if jobs <= 1:
        return [one(j) for j in todo]
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(one, todo))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cf-bench", description=__doc__)
    ap.add_argument("space", type=Path)
    ap.add_argument("--methods", default="bsa,bsa-feas,bsa-diag,random,tpe,smac,incumbent,coord-ascent")
    ap.add_argument("--seeds", default="0-9", help="ranges and comma lists, e.g. 0-9 or 0,2,5-7")
    ap.add_argument("--budget", type=int, default=1000, help="unique BLIS runs per (method, seed)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1, help="(method, seed) pairs run in parallel")
    ap.add_argument("--workers", type=int, default=None, help="BSA/random batch parallelism")
    ap.add_argument("--transfer-from", default=None, help="BSA trace JSON whose last P seeds bsa-transfer")
    a = ap.parse_args(argv)
    sp = load_space(a.space)
    runner = CachedRunner(BlisEnv.from_env(), sp.scenario, a.cache)
    opts = {k: v for k, v in (("workers", a.workers), ("transfer_from", a.transfer_from)) if v is not None}
    ts = bench(sp, runner, [m.strip() for m in a.methods.split(",") if m.strip()], parse_seeds(a.seeds),
               a.budget, a.out, opts, a.jobs, str(a.space))
    print(f"cf-bench: {len(ts)} new traces; {runner.unique_runs} unique BLIS runs, "
          f"{runner.cache_hits} cache hits", file=sys.stderr)


if __name__ == "__main__":
    main()
