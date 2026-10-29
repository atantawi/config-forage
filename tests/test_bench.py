import json
from pathlib import Path

import pytest

from config_forage.bench import analyze, grid, run
from config_forage.blis.runner import CachedRunner
from config_forage.feasibility import enumerate_feasible
from config_forage.search.base import Trace, TracePoint
from config_forage.space import load_space

TINY = Path(__file__).parent / "data" / "tiny-space.yaml"


def fake_blis(config, c, seed):
    """ITL grows with concurrency, slower for small instances and big batches; throughput ~ c."""
    width = config.get("tp", 1) * config.get("dp", 1)
    itl = 4 + c * (config["max_num_seqs"] / 64) / width
    tps = 100.0 * c * (1 + 0.05 * config.get("tp", 1)) * (1.1 if config.get("speculative") else 1.0) * (1 + 0.01 * seed)
    return {"metrics": {"itl_p99_ms": itl, "ttft_p99_ms": 50.0, "tokens_per_sec": tps, "preemption_count": 0},
            "error": None, "returncode": 0, "deployment": {"gpus": 16}}


@pytest.fixture
def setup(tmp_path):
    sp = load_space(TINY)
    return sp, CachedRunner(None, sp.scenario, tmp_path / "cache", run_fn=fake_blis), tmp_path


def trace(runs_scores, method="m", seed=0):
    pts = [TracePoint(r, s, None, i + 1) for i, (r, s) in enumerate(runs_scores)]
    return Trace(pts, method, seed)


def test_grid_resumes(setup):
    sp, runner, tmp = setup
    out = tmp / "grid.jsonl"
    s1 = grid.grid(sp, runner, out, workers=4, log=None)
    n = len(list(enumerate_feasible(sp)))
    assert s1["new"] == n and len(out.read_text().splitlines()) == n
    row = json.loads(out.read_text().splitlines()[0])
    assert {"decision", "config", "score", "c_star", "c_use", "tps_per_gpu", "runs", "diagnostics"} <= set(row)
    out.write_text(out.read_text() + '{"decision": [0, 0')  # a line cut short by a kill
    s2 = grid.grid(sp, runner, out, workers=4, log=None)
    assert s2["new"] == 0 and len(out.read_text().splitlines()) == n + 1
    assert s2["optimum"] == s1["optimum"] == analyze.optimum(analyze.load_grid(out))


def test_each_arm_gets_fresh_objective(setup):
    sp, runner, tmp = setup
    grid.grid(sp, runner, tmp / "grid.jsonl", workers=4, log=None)  # warm the cache
    before = runner.unique_runs
    traces = run.bench(sp, runner, ["bsa-feas", "random"], [0], 60, tmp / "runs")
    assert runner.unique_runs == before  # every run was a cache hit
    for t in traces:
        assert t.points and 0 < t.points[0].runs < 60 and t.points[-1].runs >= 60
        assert t.extra["stop"] == "budget"
    assert (tmp / "runs" / "random" / "seed-0.json").exists()
    m = json.loads((tmp / "runs" / "manifest.json").read_text())
    assert m["budget"] == 60 and m["arms"] == ["bsa-feas", "random"] and m["seeds"] == [0] and m["pins"]


def test_bench_resumes_and_builds_fresh_arms(setup):
    sp, runner, tmp = setup
    a = run.bench(sp, runner, ["bsa-diag"], [0, 1], 40, tmp / "runs")
    assert len(a) == 2 and a[0].to_json() != a[1].to_json()
    assert run.bench(sp, runner, ["bsa-diag"], [0, 1], 40, tmp / "runs") == []
    with pytest.raises(ValueError, match="budget"):
        run.bench(sp, runner, ["bsa-diag"], [0], 50, tmp / "runs")
    with pytest.raises(ValueError, match="transfer_from"):
        run.bench(sp, runner, ["bsa-transfer"], [0], 40, tmp / "runs")
    t = run.bench(sp, runner, ["bsa-transfer"], [0], 40, tmp / "runs",
                  opts={"transfer_from": tmp / "runs" / "bsa-diag" / "seed-0.json"})
    assert t[0].method == "bsa-transfer"


def test_parse_seeds():
    assert run.parse_seeds("0-3,7,9-10") == [0, 1, 2, 3, 7, 9, 10]


def test_trace_value_at_budget():
    t = trace([(100, -1.0), (996, -2.0), (1012, -3.0)])
    assert t.value_at(1000) == -2.0
    assert analyze.regret_table([t], -4.0, budgets=(1000,))["m"][1000]["median"] == pytest.approx(2.0)


def test_runs_to_within_negative_scores():
    t = trace([(10, None), (20, -90.0), (35, -99.5), (50, -100.0)])
    assert analyze.runs_to_within(t, -100.0) == 35  # -99.5 <= -100 + 1
    assert analyze.runs_to_within(t, -100.0, tol=0.001) == 50


def test_runs_to_within_ignores_points_past_budget():
    t = trace([(500, -90.0), (1012, -100.0)])
    assert analyze.runs_to_within(t, -100.0, budget=1000) is None
    assert analyze.runs_to_within(t, -100.0, budget=1012) == 1012


def test_bench_caps_workers_under_jobs(setup, monkeypatch):
    sp, runner, tmp = setup
    monkeypatch.setattr(run.os, "cpu_count", lambda: 9)
    run.bench(sp, runner, ["random"], [0, 1], 20, tmp / "runs", jobs=2)
    m = json.loads((tmp / "runs" / "manifest.json").read_text())
    assert m["jobs"] == 2 and m["workers"] == 4


def test_runs_to_within_censored():
    assert analyze.runs_to_within(trace([(10, None), (20, -90.0)]), -100.0) is None


def test_regret_table_shapes():
    ts = [trace([(10, -50.0), (60, -95.0)], "a", s) for s in range(4)] + [trace([(10, None)], "b", 0)]
    tab = analyze.regret_table(ts, -100.0, budgets=(50, 100))
    assert set(tab) == {"a", "b"} and set(tab["a"]) == {50, 100}
    cell = tab["a"][100]
    assert set(cell) >= {"median", "q1", "q3", "n", "n_none"}
    assert cell["median"] == pytest.approx(5.0) and cell["n"] == 4
    assert tab["b"][50] == {"median": 100.0, "q1": 100.0, "q3": 100.0, "n": 1, "n_none": 1}


def test_wilcoxon():
    assert analyze.wilcoxon([1, 2, 3], [1, 2, 3]) == 1.0
    p = analyze.wilcoxon([float(i) for i in range(10)], [i + 5.0 for i in range(10)])
    assert 0 < p < 0.05
    assert analyze.wilcoxon([None, 5], [9, 5], budget=10) == analyze.wilcoxon([11, 5], [9, 5])


def test_rescore_heldout(setup):
    sp, runner, _ = setup
    cfg = sp.default_config()
    (r,) = analyze.rescore_heldout([cfg], runner)
    assert r["config"] == cfg and len(r["scores"]) == 3 and r["mean"] == pytest.approx(sum(r["scores"]) / 3)
    assert len(set(r["scores"])) == 3


def test_analyze_writes_outputs(setup):
    sp, runner, tmp = setup
    grid.grid(sp, runner, tmp / "grid.jsonl", workers=4, log=None)
    run.bench(sp, runner, ["bsa", "random"], [0, 1, 2], 60, tmp / "runs")
    analyze.main([str(tmp / "runs"), "--grid", str(tmp / "grid.jsonl"), "--out", str(tmp / "out")])
    for f in ("regret.png", "runs_to_1pct.png", "table.md"):
        assert (tmp / "out" / f).stat().st_size > 0
    md = (tmp / "out" / "table.md").read_text()
    assert "random" in md and "vs bsa" in md
    analyze.main([str(tmp / "runs"), "--out", str(tmp / "out2")])  # no grid: best-found
    assert (tmp / "out2" / "table.md").exists() and not (tmp / "out2" / "runs_to_1pct.png").exists()
