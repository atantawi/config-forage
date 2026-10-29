"""cf-grid: probe every feasible decision once; one JSON line per config, resumable by decision.

A sidecar OUT.meta.json holds the space_hash and the BLIS pins; a resume into a grid whose sidecar
differs in either, or that has rows but no sidecar, is refused. The sidecar also records, as
information only, the git_sha of the first call and {git_sha, dirty, time} of every call."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import config_forage.bench as _pkg
from config_forage.bench.analyze import load_grid
from config_forage.blis.runner import PINS, BlisEnv, CachedRunner, run_batch
from config_forage.feasibility import enumerate_feasible
from config_forage.probe import probe
from config_forage.space import KnobSpace, load_space, space_hash


def _key(d) -> str:
    return json.dumps(list(d))


def meta_path(out: Path) -> Path:
    """The sidecar of grid file `out`: <out>.meta.json."""
    out = Path(out)
    return out.with_name(out.name + ".meta.json")


def _check_meta(space: KnobSpace, out: Path, rows: int) -> dict:
    """The sidecar to write for this call; raise if an existing grid's sidecar differs in space_hash
    or pins, or is missing while the grid has rows. git_sha is not checked."""
    want, mp, have = {"space_hash": space_hash(space), "pins": list(PINS)}, meta_path(out), {}
    if mp.exists():
        have = json.loads(mp.read_text())
        bad = [k for k in want if have.get(k) != want[k]]
        if bad:
            raise ValueError(f"{out} was probed with other {' and '.join(bad)}: "
                             + "; ".join(f"{k} {have.get(k)} there, {want[k]} now" for k in bad)
                             + "; use a fresh --out")
    elif rows:
        raise ValueError(f"{out} holds {rows} rows but no sidecar {mp.name}, so its space and pins are "
                         f"unknown (this space_hash is {want['space_hash']}); use a fresh --out")
    git = _pkg.git_state()
    return {**want, "git_sha": have.get("git_sha") if have else git["git_sha"],  # legacy sidecar: unknown
            "calls": [*have.get("calls", []), {**git, "time": _pkg.now()}]}


def grid(space: KnobSpace, runner, out: Path, workers: int | None = None, log=sys.stderr) -> dict:
    """Probe the feasible decisions not yet in `out` (appended as they finish); return a summary:
    n, new, feasible_scored, optimum, optimum_config, unique_runs and cache_hits (this call), wall_s."""
    t0, out = time.monotonic(), Path(out)
    # a line cut short by a kill is skipped and its config re-probed (the cache makes that free)
    done = {_key(r["decision"]): r for r in load_grid(out)} if out.exists() else {}
    meta = _check_meta(space, out, len(done))
    todo = [d for d in enumerate_feasible(space) if _key(d) not in done]
    runs0, hits0 = runner.unique_runs, runner.cache_hits
    lock, count = threading.Lock(), [0]
    step = max(1, len(todo) // 20)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path(out).write_text(json.dumps(meta, indent=2))
    with out.open("a") as f:
        if out.stat().st_size and not out.read_bytes().endswith(b"\n"):
            f.write("\n")  # end a line cut short by a kill before appending
        def one(d):
            p = probe(runner, space.config(d), runner.scenario)
            row = {"decision": list(d), "config": space.config(d), "score": p.score, "c_star": p.c_star,
                   "c_use": p.c_use, "tps_per_gpu": p.tps_per_gpu, "runs": p.runs, "diagnostics": p.diagnostics}
            with lock:
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
                done[_key(d)] = row
                count[0] += 1
                if log is not None and (count[0] % step == 0 or count[0] == len(todo)):
                    print(f"cf-grid: {count[0]}/{len(todo)} ({100 * count[0] // len(todo)}%), "
                          f"{runner.unique_runs - runs0} BLIS runs, {time.monotonic() - t0:.0f}s", file=log)
            return row

        run_batch(one, todo, workers)
    ok = [r for r in done.values() if r["score"] is not None]
    best = min(ok, key=lambda r: r["score"]) if ok else None
    return {"n": len(done), "new": len(todo), "feasible_scored": len(ok),
            "optimum": best and best["score"], "optimum_config": best and best["config"],
            "unique_runs": runner.unique_runs - runs0, "cache_hits": runner.cache_hits - hits0,
            "wall_s": time.monotonic() - t0}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cf-grid", description=__doc__)
    ap.add_argument("space", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    sp = load_space(a.space)
    s = grid(sp, CachedRunner(BlisEnv.from_env(), sp.scenario, a.cache), a.out, a.workers)
    print(f"cf-grid: {s['n']} configs ({s['new']} new), {s['feasible_scored']} feasible-scored; "
          f"optimum {s['optimum']} at {s['optimum_config']}; {s['unique_runs']} unique BLIS runs, "
          f"{s['cache_hits']} cache hits; {s['wall_s']:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
