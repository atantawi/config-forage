"""cf-analyze: regret vs budget, runs-to-within-1%, paired Wilcoxon tests, held-out rescoring.

Scores are negative (-tokens/s per GPU), lower is better; optimum = min grid score. Regret at budget
B = trace.value_at(B) - optimum >= 0, read at the last trace point with runs <= B; a trace with no
success by B counts as score 0 (regret |optimum|) and is tallied in n_none.

TPE traces also appear re-indexed as "tpe (pruned = 1 run)": the other cost convention, where each
pruned (infeasible) suggestion costs one run. --heldout N rescores each method's top N distinct
configs on held-out workload seeds (needs the BLIS environment)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from config_forage.probe import probe
from config_forage.search.base import Trace, TracePoint

BUDGETS = (50, 100, 200, 500, 1000)
TIE_TOL = 0.005  # upstream's tie tolerance: within 0.5% of the optimum counts as a tie
HELDOUT_SEEDS = (1, 2, 3)
PRUNED_SUFFIX = " (pruned = 1 run)"


def load_traces(d: str | Path) -> list[Trace]:
    """Every <arm>/seed-<k>.json under d, sorted by (method, seed)."""
    ts = [Trace.from_json(p.read_text()) for p in sorted(Path(d).glob("*/seed-*.json"))]
    return sorted(ts, key=lambda t: (t.method, t.seed))


def load_grid(path: str | Path) -> list[dict]:
    """grid.jsonl rows; blank or truncated lines (a killed cf-grid) are skipped."""
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def optimum(rows: Sequence[dict]) -> float:
    scores = [r["score"] for r in rows if r["score"] is not None]
    if not scores:
        raise ValueError("no scored config in the grid")
    return min(scores)


def check_not_below(traces: Sequence[Trace], optimum: float) -> None:
    """Raise if any trace's best is below the grid optimum by more than 1e-9·|optimum|: the trace
    then scored a config the grid does not contain (a feasibility leak), and regret is meaningless."""
    bad = [(t.method, t.seed, p.best_score) for t in traces for p in t.points[-1:]
           if p.best_score is not None and p.best_score < optimum - 1e-9 * abs(optimum)]
    if bad:
        raise ValueError(f"trace best below the grid optimum {optimum!r} (a config outside the grid's "
                         f"feasible space was scored): {bad[:5]}{' ...' if len(bad) > 5 else ''}")


def runs_to_within(trace: Trace, optimum: float, budget: int, tol: float = 0.01) -> int | None:
    """Runs at the first point whose best score is within tol·|optimum| of the optimum; None if no
    point with runs <= budget gets there (points a batch overshot past the budget never count)."""
    bound = optimum + tol * abs(optimum)
    for p in trace.points:
        if p.runs > budget:
            return None
        if p.best_score is not None and p.best_score <= bound:
            return p.runs
    return None


def _by_method(traces: Sequence[Trace]) -> dict[str, list[Trace]]:
    out: dict[str, list[Trace]] = defaultdict(list)
    for t in traces:
        out[t.method].append(t)
    return dict(out)


def identical_arms(traces: Sequence[Trace]) -> list[tuple[str, str]]:
    """Pairs of methods (sorted) whose traces are identical seed for seed (same seeds, same points):
    an arm whose biaser never acted, say. The trace's extra (e.g. BSA history) is not compared."""
    by = {m: {t.seed: t.points for t in ts} for m, ts in _by_method(traces).items()}
    ms = sorted(by)
    return [(a, b) for i, a in enumerate(ms) for b in ms[i + 1:] if by[a] == by[b]]


def _summary(xs: Sequence[float], n_none: int) -> dict:
    q1, med, q3 = np.percentile(xs, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "n": len(xs), "n_none": n_none}


def regrets(trace: Trace, optimum: float, budget: int) -> float | None:
    v = trace.value_at(budget)
    return None if v is None else v - optimum


def regret_table(traces: Sequence[Trace], optimum: float, budgets: Sequence[int] = BUDGETS) -> dict:
    """{method: {budget: {median, q1, q3, n, n_none}}} of regret over seeds (None -> |optimum|)."""
    tab = {}
    for m, ts in _by_method(traces).items():
        tab[m] = {}
        for b in budgets:
            rs = [regrets(t, optimum, b) for t in ts]
            tab[m][b] = _summary([abs(optimum) if r is None else r for r in rs], sum(r is None for r in rs))
    return tab


def best_table(traces: Sequence[Trace], budgets: Sequence[int] = BUDGETS) -> dict:
    """As regret_table but of the best score found (no grid); None -> 0 (zero throughput)."""
    tab = {}
    for m, ts in _by_method(traces).items():
        tab[m] = {}
        for b in budgets:
            vs = [t.value_at(b) for t in ts]
            tab[m][b] = _summary([0.0 if v is None else v for v in vs], sum(v is None for v in vs))
    return tab


def wilcoxon(a: Sequence[float | None], b: Sequence[float | None], budget: int | None = None) -> float:
    """Two-sided paired Wilcoxon p-value (zero_method="zsplit"); None entries (censored
    runs-to-within) become budget + 1. 1.0 if every difference is zero; NaN for an empty pairing."""
    from scipy.stats import wilcoxon as _w

    def fill(xs):
        if any(x is None for x in xs) and budget is None:
            raise ValueError("censored values need the budget")
        return np.array([budget + 1 if x is None else x for x in xs], dtype=float)

    x, y = fill(a), fill(b)
    if len(x) != len(y):
        raise ValueError(f"unpaired samples: {len(x)} vs {len(y)}")
    if len(x) == 0:
        return float("nan")
    if not np.any(x - y):
        return 1.0
    return float(_w(x, y, zero_method="zsplit", alternative="two-sided").pvalue)


def rescore_heldout(top_configs: Sequence[dict], runner, seeds: Sequence[int] = (1, 2, 3)) -> list[dict]:
    """Probe each config at each held-out workload seed: [{config, scores, mean, failures, mean_ok}].
    A failed seed counts as score 0.0 (zero throughput, the worst) in `mean`, so a config that fails
    on some seeds cannot outrank one that succeeds on all; `mean_ok` is the mean over the successes
    (None if none) and `failures` the count of failed seeds."""
    out = []
    for cfg in top_configs:
        scores = [probe(runner, cfg, runner.scenario, s).score for s in seeds]
        ok = [s for s in scores if s is not None]
        out.append({"config": cfg, "scores": scores,
                    "mean": float(np.mean([0.0 if s is None else s for s in scores])),
                    "failures": len(scores) - len(ok), "mean_ok": float(np.mean(ok)) if ok else None})
    return out


def pruned_as_one_run(trace: Trace) -> Trace:
    """A TPE trace under the other cost convention: every pruned trial adds 1 to the runs of all
    later points. Uses extra["pruned_after"] (points recorded before each prune); without it, a
    prune at runs_used q is taken to precede the points with runs > q."""
    after = trace.extra.get("pruned_after")
    at = trace.extra.get("pruned_at", [])
    pts = []
    for i, p in enumerate(trace.points):
        k = sum(a <= i for a in after) if after is not None else sum(q < p.runs for q in at)
        pts.append(TracePoint(p.runs + k, p.best_score, p.best_config, p.evaluated, p.decision, p.score))
    return Trace(pts, trace.method + PRUNED_SUFFIX, trace.seed, {"derived_from": trace.method})


def with_pruned_rows(traces: Sequence[Trace]) -> list[Trace]:
    """traces plus a re-indexed copy of each trace that logged prunes (TPE)."""
    return [*traces, *(pruned_as_one_run(t) for t in traces if "pruned_at" in t.extra)]


def top_configs(ts: Sequence[Trace], n: int, space) -> list[tuple[float, dict]]:
    """A method's n best distinct successful configs across its seeds, (search score, config), best
    first. Traces from before per-evaluation records fall back to their best_config points."""
    best: dict[str, tuple[float, dict]] = {}
    for t in ts:
        cands = t.top_configs(len(t.points), space) if any(p.decision is not None for p in t.points) else \
            [(p.best_score, p.best_config) for p in t.points if p.best_score is not None and p.best_config]
        for s, cfg in cands:
            k = json.dumps(cfg, sort_keys=True, default=str)
            if k not in best or s < best[k][0]:
                best[k] = (s, cfg)
    return sorted(best.values(), key=lambda x: x[0])[:n]


def heldout(runs_dir: Path, out: Path, n: int, runner, space, seeds: Sequence[int] = HELDOUT_SEEDS) -> list[dict]:
    """Rescore each method's top-n distinct configs on held-out workload seeds; write heldout.md.
    Rows: {method, config, search_score, scores, mean, failures, mean_ok, tie}; a failed seed counts
    as 0.0 in mean (see rescore_heldout). A row is a tie when it has no failure and its mean is within
    0.5% of the best mean over all failure-free rows; a row with a failure is never a tie or winner."""
    by = _by_method(load_traces(runs_dir))
    rows = []
    for m in sorted(by):
        top = top_configs(by[m], n, space)
        for (s, _), r in zip(top, rescore_heldout([c for _, c in top], runner, seeds)):
            rows.append({"method": m, "search_score": s, **r})
    clean = [r["mean"] for r in rows if r["failures"] == 0]
    best = min(clean) if clean else None
    for r in rows:
        r["tie"] = best is not None and r["failures"] == 0 and r["mean"] <= best + TIE_TOL * abs(best)
    lines = [f"# Held-out rescoring: top {n} distinct configs per method, seeds {tuple(seeds)}", "",
             "Score = -tokens/s per GPU (lower is better). held-out mean counts a failed seed as 0 (zero "
             "throughput); mean over successes ignores failures and is shown for reference only. "
             + (f"tie = no failure and held-out mean within 0.5% of the best failure-free mean ({best:.6g}); "
                "a config with a failure is never a tie." if best is not None
                else "No config succeeded on every held-out seed, so none is a tie."), "",
             "| method | rank | search score | held-out mean | failures | mean over successes | held-out scores "
             "| tie | config |",
             "|---|---|---|---|---|---|---|---|---|"]
    rank: dict[str, int] = defaultdict(int)
    for r in rows:
        rank[r["method"]] += 1
        sc = ", ".join("fail" if x is None else f"{x:.6g}" for x in r["scores"])
        mean_ok = "fail" if r["mean_ok"] is None else f"{r['mean_ok']:.6g}"
        cfg = json.dumps(r["config"], default=str)
        lines.append(f"| {r['method']} | {rank[r['method']]} | {r['search_score']:.6g} | {r['mean']:.6g} | "
                     f"{r['failures']} | {mean_ok} | {sc} | {'tie' if r['tie'] else ''} | `{cfg}` |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout.md").write_text("\n".join(lines) + "\n")
    return rows


def _paired(ts_a: list[Trace], ts_b: list[Trace], f) -> tuple[list, list]:
    a, b = {t.seed: f(t) for t in ts_a}, {t.seed: f(t) for t in ts_b}
    common = sorted(set(a) & set(b))
    return [a[s] for s in common], [b[s] for s in common]


def _fmt(c: dict) -> str:
    s = f"{c['median']:.3g} [{c['q1']:.3g}, {c['q3']:.3g}]"
    return s + (f" ({c['n_none']} none)" if c["n_none"] else "")


def _plot_curves(by: dict[str, list[Trace]], f, ylabel: str, path: Path, max_runs: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.unique(np.geomspace(1, max(2, max_runs), 200).astype(int))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, ts in sorted(by.items()):
        ys = np.array([[f(t, int(x)) for x in xs] for t in ts])
        med, q1, q3 = np.percentile(ys, [50, 25, 75], axis=0)
        ax.plot(xs, med, label=m)
        ax.fill_between(xs, q1, q3, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("unique BLIS runs")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_runs_to(by: dict[str, list[Trace]], opt: float, budget: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted(by)
    data = [[budget + 1 if (r := runs_to_within(t, opt, budget)) is None else r for t in by[m]] for m in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, showfliers=False)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, 1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=10, color="k", alpha=0.6)
    ax.set_xticks(range(1, len(names) + 1), names, rotation=30, ha="right")
    ax.axhline(budget + 1, color="grey", lw=0.8, ls="--")
    ax.set_ylabel(f"unique BLIS runs to within 1% (censored = {budget + 1})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def check_space_match(runs_dir: Path, grid_path: Path, log=None) -> bool:
    """Compare the space_hash of the runs manifest and of the grid sidecar (<grid>.meta.json): regret
    against another space's optimum is meaningless. Raise ValueError, naming both hashes, when both
    exist and differ. Warn (to log) and return False when either is missing (legacy metadata, so
    the analysis goes on); True when they agree."""
    def read(p: Path):
        try:
            return json.loads(p.read_text()).get("space_hash")
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    log = sys.stderr if log is None else log
    runs_h = read(Path(runs_dir) / "manifest.json")
    grid_h = read(Path(grid_path).with_name(Path(grid_path).name + ".meta.json"))
    if runs_h is None or grid_h is None:
        which = " and ".join(w for w, h in (("runs manifest", runs_h), ("grid sidecar", grid_h)) if h is None)
        print(f"cf-analyze: warning: space_hash missing from the {which}; cannot check that the runs and "
              f"the grid are of the same space", file=log)
        return False
    if runs_h != grid_h:
        raise ValueError(f"space_hash of the runs ({runs_h}, {runs_dir}) and of the grid ({grid_h}, {grid_path}) "
                         f"differ: they are of different spaces, so regret would be against another space's "
                         f"optimum; analyze the runs against their own space's grid")
    return True


def provenance_note(traces: Sequence[Trace]) -> str | None:
    """A one-line caveat when the traces were written by more than one git revision, or by a dirty
    checkout (extra["provenance"], set by cf-bench); None when they agree and are clean, or carry none."""
    provs = [t.extra.get("provenance") for t in traces]
    shas = sorted({str(p.get("git_sha")) for p in provs if p}, key=str)
    dirty = sum(bool(p and p.get("dirty")) for p in provs)
    legacy = sum(p is None for p in provs)
    if len(shas) <= 1 and not dirty:
        return None
    parts = []
    if len(shas) > 1:
        parts.append(f"the traces span {len(shas)} git revisions ({', '.join(s[:12] for s in shas)})")
    if dirty:
        parts.append(f"{dirty} of {len(provs)} traces were written from a dirty checkout")
    if legacy:
        parts.append(f"{legacy} carry no provenance")
    return "; ".join(parts) + ", so they may not all come from the same code"


def analyze(runs_dir: Path, out: Path, grid_path: Path | None = None, budgets: Sequence[int] = BUDGETS) -> str:
    """Write regret.png (best found without a grid), runs_to_1pct.png (grid only) and table.md."""
    raw = load_traces(runs_dir)
    if not raw:
        raise ValueError(f"no traces under {runs_dir}")
    if grid_path is not None:
        check_space_match(runs_dir, grid_path)  # before any output: a definite mismatch raises
    mpath = Path(runs_dir) / "manifest.json"
    budget = (json.loads(mpath.read_text()).get("budget") if mpath.exists() else None) or \
        max(p.runs for t in raw for p in t.points)
    traces = with_pruned_rows(raw)
    by = _by_method(traces)
    out.mkdir(parents=True, exist_ok=True)
    budgets = [b for b in budgets if b <= budget] or [budget]
    lines = [f"# Benchmark: {len(by)} methods, budget {budget} unique BLIS runs", ""]
    note = provenance_note(raw)
    if note:
        print(f"cf-analyze: warning: {note}", file=sys.stderr)
        lines += [f"**Provenance:** {note}.", ""]
    refs = [r for r in ("bsa", "tpe") if r in by]
    if grid_path is not None:
        opt = optimum(load_grid(grid_path))
        check_not_below(raw, opt)
        tab = regret_table(traces, opt, budgets)
        lines += [(f"Grid optimum: {opt:.6g} (score = -tokens/s per GPU). Regret = best - optimum, "
                   f"median [IQR] over seeds; no success counts as regret {abs(opt):.6g}."), ""]

        def val(t, b):
            r = regrets(t, opt, b)
            return abs(opt) if r is None else r

        _plot_curves(by, val, "regret (tokens/s per GPU)", out / "regret.png", budget)
        _plot_runs_to(by, opt, budget, out / "runs_to_1pct.png")
    else:
        tab = best_table(traces, budgets)
        lines += ["No grid: best score found (-tokens/s per GPU, lower is better), median [IQR] over seeds.", ""]

        def val(t, b):
            v = t.value_at(b)
            return 0.0 if v is None else v

        _plot_curves(by, val, "best score (-tokens/s per GPU)", out / "regret.png", budget)
    same = identical_arms(raw)
    if same:
        lines += ["**Identical arms** (traces equal seed for seed, so they are one arm here): "
                  + "; ".join(f"{a}, {b}" for a, b in same) + ".", ""]
    if "incumbent" in by:
        lines += ["incumbent is deterministic (the seed is ignored): its seeds are copies of one trace, so its "
                  "n carries no extra information (zero IQR; a Wilcoxon test against it pairs one value n times).", ""]
    if any(m.endswith(PRUNED_SUFFIX) for m in by):
        lines += [f"tpe{PRUNED_SUFFIX}: the TPE traces re-indexed so that each pruned (infeasible) suggestion "
                  "costs one run; plain tpe counts them free.", ""]
    head = "| method | seeds | " + " | ".join(f"B={b}" for b in budgets)
    head += f" | runs to 1% | % seeds within 0.5% at B={budgets[-1]} |" if grid_path is not None else " |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for m in sorted(by):
        row = f"| {m} | {len(by[m])} | " + " | ".join(_fmt(tab[m][b]) for b in budgets)
        if grid_path is not None:
            rt = [runs_to_within(t, opt, budget) for t in by[m]]
            c = _summary([budget + 1 if r is None else r for r in rt], sum(r is None for r in rt))
            row += f" | {c['median']:.0f} [{c['q1']:.0f}, {c['q3']:.0f}]" + \
                (f" ({c['n_none']} censored)" if c["n_none"] else "")
            vs = [t.value_at(budgets[-1]) for t in by[m]]
            tie = sum(v is not None and v <= opt + TIE_TOL * abs(opt) for v in vs) / len(vs)
            row += f" | {100 * tie:.0f}%"
        lines.append(row + " |")
    for ref in refs:
        lines += ["", f"## Wilcoxon p-values vs {ref} (two-sided, paired by seed)", ""]
        cols = [f"B={budgets[-1]}"] + (["runs to 1%"] if grid_path is not None else [])
        lines += ["| method | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for m in sorted(by):
            if m == ref:
                continue
            pair = _paired(by[m], by[ref], lambda t: val(t, budgets[-1]))
            ps = [wilcoxon(*pair)]
            if grid_path is not None:
                ps.append(wilcoxon(*_paired(by[m], by[ref], lambda t: runs_to_within(t, opt, budget)), budget=budget))
            lines.append(f"| {m} | {len(pair[0])} | " + " | ".join(f"{p:.3g}" for p in ps) + " |")
    md = "\n".join(lines) + "\n"
    (out / "table.md").write_text(md)
    return md


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cf-analyze", description=__doc__)
    ap.add_argument("runs", type=Path)
    ap.add_argument("--grid", type=Path, default=None, help="grid.jsonl; without it, report best found")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--heldout", type=int, default=None, metavar="N",
                    help=f"also rescore each method's top N distinct configs on seeds {HELDOUT_SEEDS} "
                         "(needs --space, --cache and the BLIS_* environment)")
    ap.add_argument("--space", type=Path, default=None, help="space YAML (for --heldout)")
    ap.add_argument("--cache", type=Path, default=None, help="BLIS cache dir (for --heldout)")
    a = ap.parse_args(argv)
    if a.heldout is not None and (a.space is None or a.cache is None):
        ap.error("--heldout needs --space and --cache")
    analyze(a.runs, a.out, a.grid)
    if a.heldout is not None:
        from config_forage.blis.runner import BlisEnv, CachedRunner
        from config_forage.space import load_space
        sp = load_space(a.space)
        heldout(a.runs, a.out, a.heldout, CachedRunner(BlisEnv.from_env(), sp.scenario, a.cache), sp)


if __name__ == "__main__":
    main()
