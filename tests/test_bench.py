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
    src = {"transfer_from": tmp / "runs" / "bsa-diag" / "seed-0.json"}
    with pytest.raises(ValueError, match="opts"):  # resuming with other options would mix runs
        run.bench(sp, runner, ["bsa-transfer"], [0], 40, tmp / "runs", opts=src)
    t = run.bench(sp, runner, ["bsa-transfer"], [0], 40, tmp / "runs-t", opts=src)
    assert t[0].method == "bsa-transfer"
    assert run.bench(sp, runner, ["bsa-transfer"], [0], 40, tmp / "runs-t", opts=src, jobs=2) == []


def test_parse_seeds():
    assert run.parse_seeds("0-3,7,9-10") == [0, 1, 2, 3, 7, 9, 10]


def test_trace_value_at_budget():
    t = trace([(100, -1.0), (996, -2.0), (1012, -3.0)])
    assert t.value_at(1000) == -2.0
    assert analyze.regret_table([t], -4.0, budgets=(1000,))["m"][1000]["median"] == pytest.approx(2.0)


def test_runs_to_within_negative_scores():
    t = trace([(10, None), (20, -90.0), (35, -99.5), (50, -100.0)])
    assert analyze.runs_to_within(t, -100.0, 1000) == 35  # -99.5 <= -100 + 1
    assert analyze.runs_to_within(t, -100.0, 1000, tol=0.001) == 50


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
    assert analyze.runs_to_within(trace([(10, None), (20, -90.0)]), -100.0, 1000) is None


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


def test_analyze_fails_loudly_below_grid_optimum(setup):
    sp, runner, tmp = setup
    grid.grid(sp, runner, tmp / "grid.jsonl", workers=4, log=None)
    opt = analyze.optimum(analyze.load_grid(tmp / "grid.jsonl"))
    d = tmp / "runs" / "m"
    d.mkdir(parents=True)
    (d / "seed-0.json").write_text(trace([(5, opt * 1.01)]).to_json())
    with pytest.raises(ValueError, match="below the grid optimum"):
        analyze.analyze(tmp / "runs", tmp / "out", tmp / "grid.jsonl")
    (d / "seed-0.json").write_text(trace([(5, opt * (1 + 1e-12))]).to_json())  # within 1e-9·|opt|
    analyze.analyze(tmp / "runs", tmp / "out", tmp / "grid.jsonl")


def test_bench_manifest_records_active_rules(setup, capsys):
    sp, runner, tmp = setup  # tiny space: max_num_seqs has 3 values, no TTFT SLO
    run.bench(sp, runner, ["bsa-diag", "random"], [0], 20, tmp / "runs")
    m = json.loads((tmp / "runs" / "manifest.json").read_text())
    assert m["active_rules"] == {"bsa-diag": [{"signal": "kv_starved", "knob": "max_num_seqs", "prefer": "lower"},
                                              {"signal": "itl", "knob": "max_num_seqs", "prefer": "lower"}]}
    assert "no active" not in capsys.readouterr().err
    pinned = load_space(TINY)
    from config_forage.space import Knob, KnobSpace
    pinned = KnobSpace(pinned.scenario, tuple(Knob(k.name, k.values[:1]) if k.name == "max_num_seqs" else k
                                              for k in pinned.knobs))
    run.bench(pinned, runner, ["bsa-diag"], [0], 20, tmp / "runs2")
    assert "bsa-diag: no active diagnostic rule" in capsys.readouterr().err
    assert json.loads((tmp / "runs2" / "manifest.json").read_text())["active_rules"] == {"bsa-diag": []}


def test_analyze_flags_identical_arms(setup):
    _, _, tmp = setup
    for m in ("a", "b", "c"):
        (tmp / "runs" / m).mkdir(parents=True)
        for s in (0, 1):
            pts = [(5, -1.0 - s), (9, -2.0)] if m != "c" else [(5, -1.0), (8, -2.0)]
            (tmp / "runs" / m / f"seed-{s}.json").write_text(trace(pts, m, s).to_json())
    assert analyze.identical_arms(analyze.load_traces(tmp / "runs")) == [("a", "b")]
    md = analyze.analyze(tmp / "runs", tmp / "out")
    assert "Identical arms" in md and "a, b" in md


def test_heldout_rescoring_with_fake_runner(setup):
    sp, runner, tmp = setup
    run.bench(sp, runner, ["bsa-feas", "random"], [0, 1], 40, tmp / "runs")
    rows = analyze.heldout(tmp / "runs", tmp / "out", 3, runner, sp)
    assert {r["method"] for r in rows} == {"bsa-feas", "random"}
    for m in ("bsa-feas", "random"):
        mine = [r for r in rows if r["method"] == m]
        assert len(mine) == 3 and len({json.dumps(r["config"], sort_keys=True) for r in mine}) == 3
        assert [r["search_score"] for r in mine] == sorted(r["search_score"] for r in mine)
        assert all(len(r["scores"]) == 3 for r in mine)
        # fake_blis scales throughput by (1 + 0.01·seed): held-out seeds 1..3 score better than seed 0
        assert all(r["mean"] < r["search_score"] for r in mine)
    md = (tmp / "out" / "heldout.md").read_text()
    assert "bsa-feas" in md and "seeds (1, 2, 3)" in md and "within 0.5%" in md


def test_table_tie_column_and_paired_n(setup):
    _, _, tmp = setup
    (tmp / "g.jsonl").write_text(json.dumps({"decision": [0], "config": {}, "score": -100.0}) + "\n")
    for m, finals in (("a", [-100.0, -99.6, -99.0, -90.0]), ("bsa", [-100.0] * 3)):
        (tmp / "runs" / m).mkdir(parents=True)
        for s, v in enumerate(finals):
            (tmp / "runs" / m / f"seed-{s}.json").write_text(trace([(5, -50.0), (10, v)], m, s).to_json())
    md = analyze.analyze(tmp / "runs", tmp / "out", tmp / "g.jsonl", budgets=(10,))
    assert "% seeds within 0.5% at B=10" in md
    row = next(line for line in md.splitlines() if line.startswith("| a |"))
    assert "| 50% |" in row  # -100 and -99.6 are within 0.5%; -99 and -90 are not
    assert "| n |" in md and "| a | 3 |" in md.split("Wilcoxon")[1]  # paired over the 3 common seeds


def test_wilcoxon_empty_pairing_is_nan():
    import math
    assert math.isnan(analyze.wilcoxon([], []))


def test_runs_to_within_requires_budget():
    with pytest.raises(TypeError):
        analyze.runs_to_within(trace([(1, -1.0)]), -1.0)


def test_pruned_as_one_run_reindex():
    t = trace([(2, -1.0), (4, -2.0), (4, -2.0), (7, -3.0)], "tpe")
    t.extra.update({"pruned_at": [0, 4, 4, 7], "pruned_after": [0, 2, 2, 4]})
    r = analyze.pruned_as_one_run(t)
    assert r.method == "tpe (pruned = 1 run)" and r.seed == t.seed
    assert [p.runs for p in r.points] == [3, 5, 7, 10]  # +1 before all; +2 after the 2nd point
    assert [p.best_score for p in r.points] == [p.best_score for p in t.points]
    old = trace([(2, -1.0), (5, -2.0)], "tpe")
    old.extra["pruned_at"] = [2, 3]  # no pruned_after: a prune at runs q precedes points with runs > q
    assert [p.runs for p in analyze.pruned_as_one_run(old).points] == [2, 7]


def test_analyze_adds_tpe_pruned_row(setup):
    _, _, tmp = setup
    (tmp / "g.jsonl").write_text(json.dumps({"decision": [0], "config": {}, "score": -100.0}) + "\n")
    (tmp / "runs" / "tpe").mkdir(parents=True)
    for s in range(2):
        t = trace([(5, -90.0), (10, -100.0)], "tpe", s)
        t.extra.update({"pruned_at": [5] * 6, "pruned_after": [1] * 6})
        (tmp / "runs" / "tpe" / f"seed-{s}.json").write_text(t.to_json())
    md = analyze.analyze(tmp / "runs", tmp / "out", tmp / "g.jsonl", budgets=(10,))
    row = next(line for line in md.splitlines() if line.startswith("| tpe (pruned = 1 run) |"))
    assert "| 10 [10, 10] |" in row  # at B=10 the re-indexed trace (runs 16) has only -90: regret 10


def test_manifest_counts_timeouts_per_call(setup):
    sp, _, tmp = setup
    r = CachedRunner(None, sp.scenario, tmp / "c2",
                     run_fn=lambda cfg, c, s: {**fake_blis(cfg, c, s), "returncode": -1 if c == 64 else 0})
    run.bench(sp, r, ["random"], [0], 20, tmp / "runs")
    run.bench(sp, r, ["random"], [1], 20, tmp / "runs")
    calls = json.loads((tmp / "runs" / "manifest.json").read_text())["calls"]
    assert len(calls) == 2 and all({"new_traces", "unique_runs", "cache_hits", "timeouts"} <= set(c) for c in calls)
    assert calls[0]["new_traces"] == 1 and calls[0]["timeouts"] > 0
