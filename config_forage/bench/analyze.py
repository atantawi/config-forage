"""cf-analyze: regret vs budget, runs-to-within-1%, paired Wilcoxon tests, held-out rescoring.

Scores are negative (-tokens/s per GPU), lower is better; optimum = min grid score. Regret at budget
B = trace.value_at(B) - optimum >= 0, read at the last trace point with runs <= B; a trace with no
success by B counts as score 0 (regret |optimum|) and is tallied in n_none."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from config_forage.probe import probe
from config_forage.search.base import Trace

BUDGETS = (50, 100, 200, 500, 1000)


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


def runs_to_within(trace: Trace, optimum: float, tol: float = 0.01, budget: int | None = None) -> int | None:
    """Runs at the first point whose best score is within tol·|optimum| of the optimum; None if no
    point with runs <= budget gets there (points a batch overshot past the budget never count)."""
    bound = optimum + tol * abs(optimum)
    for p in trace.points:
        if budget is not None and p.runs > budget:
            return None
        if p.best_score is not None and p.best_score <= bound:
            return p.runs
    return None


def _by_method(traces: Sequence[Trace]) -> dict[str, list[Trace]]:
    out: dict[str, list[Trace]] = defaultdict(list)
    for t in traces:
        out[t.method].append(t)
    return dict(out)


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
    runs-to-within) become budget + 1. 1.0 if every difference is zero."""
    from scipy.stats import wilcoxon as _w

    def fill(xs):
        if any(x is None for x in xs) and budget is None:
            raise ValueError("censored values need the budget")
        return np.array([budget + 1 if x is None else x for x in xs], dtype=float)

    x, y = fill(a), fill(b)
    if len(x) != len(y):
        raise ValueError(f"unpaired samples: {len(x)} vs {len(y)}")
    if not np.any(x - y):
        return 1.0
    return float(_w(x, y, zero_method="zsplit", alternative="two-sided").pvalue)


def rescore_heldout(top_configs: Sequence[dict], runner, seeds: Sequence[int] = (1, 2, 3)) -> list[dict]:
    """Probe each config at each held-out workload seed: [{config, scores, mean}] (mean over successes)."""
    out = []
    for cfg in top_configs:
        scores = [probe(runner, cfg, runner.scenario, s).score for s in seeds]
        ok = [s for s in scores if s is not None]
        out.append({"config": cfg, "scores": scores, "mean": float(np.mean(ok)) if ok else None})
    return out


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
    data = [[budget + 1 if (r := runs_to_within(t, opt, budget=budget)) is None else r for t in by[m]] for m in names]
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


def analyze(runs_dir: Path, out: Path, grid_path: Path | None = None, budgets: Sequence[int] = BUDGETS) -> str:
    """Write regret.png (best found without a grid), runs_to_1pct.png (grid only) and table.md."""
    traces = load_traces(runs_dir)
    if not traces:
        raise ValueError(f"no traces under {runs_dir}")
    mpath = Path(runs_dir) / "manifest.json"
    budget = (json.loads(mpath.read_text()).get("budget") if mpath.exists() else None) or \
        max(p.runs for t in traces for p in t.points)
    by = _by_method(traces)
    out.mkdir(parents=True, exist_ok=True)
    budgets = [b for b in budgets if b <= budget] or [budget]
    lines = [f"# Benchmark: {len(by)} methods, budget {budget} unique BLIS runs", ""]
    refs = [r for r in ("bsa", "tpe") if r in by]
    if grid_path is not None:
        opt = optimum(load_grid(grid_path))
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
    head = "| method | seeds | " + " | ".join(f"B={b}" for b in budgets)
    head += " | runs to 1% |" if grid_path is not None else " |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for m in sorted(by):
        row = f"| {m} | {len(by[m])} | " + " | ".join(_fmt(tab[m][b]) for b in budgets)
        if grid_path is not None:
            rt = [runs_to_within(t, opt, budget=budget) for t in by[m]]
            c = _summary([budget + 1 if r is None else r for r in rt], sum(r is None for r in rt))
            row += f" | {c['median']:.0f} [{c['q1']:.0f}, {c['q3']:.0f}]" + \
                (f" ({c['n_none']} censored)" if c["n_none"] else "")
        lines.append(row + " |")
    for ref in refs:
        lines += ["", f"## Wilcoxon p-values vs {ref} (two-sided, paired by seed)", ""]
        cols = [f"B={budgets[-1]}"] + (["runs to 1%"] if grid_path is not None else [])
        lines += ["| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for m in sorted(by):
            if m == ref:
                continue
            ps = [wilcoxon(*_paired(by[m], by[ref], lambda t: val(t, budgets[-1])))]
            if grid_path is not None:
                ps.append(wilcoxon(*_paired(by[m], by[ref], lambda t: runs_to_within(t, opt, budget=budget)), budget=budget))
            lines.append(f"| {m} | " + " | ".join(f"{p:.3g}" for p in ps) + " |")
    md = "\n".join(lines) + "\n"
    (out / "table.md").write_text(md)
    return md


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cf-analyze", description=__doc__)
    ap.add_argument("runs", type=Path)
    ap.add_argument("--grid", type=Path, default=None, help="grid.jsonl; without it, report best found")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    analyze(a.runs, a.out, a.grid)


if __name__ == "__main__":
    main()
