from dataclasses import replace

import pytest

from config_forage.blis.runner import CachedRunner
from config_forage.ladder import climb, rounded, servable, worst_metric
from config_forage.probe import ProbeResult, probe
from config_forage.space import Scenario, Slo

CFG = {"tp": 8, "num_instances": 1}
GPUS = 8
RUNGS = (1, 2, 4, 8)
SC = Scenario("m", "h100", "chatbot", 8, "0.29.0", scoring="ladder", concurrency=RUNGS)
NO_SLO = Slo(itl_p99_ms=None)


def rec(itl=10.0, ttft=100.0, e2e=1000.0, tps=800.0, inst=None, error=None, rc=0, gpus=GPUS, al=None,
        injected=None):
    m = {"itl_p99_ms": itl, "ttft_p99_ms": ttft, "e2e_p99_ms": e2e, "tokens_per_sec": tps}
    if injected is not None:
        m["injected_requests"] = injected
    dep = {"gpus": gpus}
    if al is not None:
        dep["acceptance"] = {"acceptance_length": al}
    return {"metrics": m, "metrics_all_instances": inst if inst is not None else [dict(m)],
            "error": error, "returncode": rc, "deployment": dep}


def runner(tmp_path, by_c, sc=SC):
    """CachedRunner over a run_fn returning by_c(c) for concurrency c; records the rungs it ran."""
    ran = []

    def fn(config, c, seed):
        ran.append(c)
        return by_c(c)
    r = CachedRunner(None, sc, tmp_path, run_fn=fn)
    r.ran = ran
    return r


class FakeRunner:
    """A plain runner: a dict of concurrency -> record; remembers each call's arguments."""

    def __init__(self, recs):
        self.recs, self.calls = recs, []

    def run(self, config, c, seed=None, shed_guard=True):
        self.calls.append((c, seed, shed_guard))
        return {**self.recs[c], "cache_key": f"k{c}"}


# --- worst_metric / servable / rounded ---------------------------------------------------------

def test_worst_metric_skips_none_and_non_numeric():
    r = rec(itl=None, inst=[{"itl_p99_ms": "x"}, {"itl_p99_ms": 7}, {}, {"itl_p99_ms": None}])
    assert worst_metric(r, "itl_p99_ms") == 7.0
    assert worst_metric(rec(itl=None, inst=[{}]), "itl_p99_ms") is None


def test_worst_instance_beats_aggregate():
    r = rec(itl=15.0, inst=[{"itl_p99_ms": 12.0}, {"itl_p99_ms": 25.0}])
    assert servable(r, Slo(20.0), GPUS) == (False, "itl_p99 25.0 ms > 20 ms")


def test_aggregate_above_instances_counts():
    r = rec(itl=25.0, inst=[{"itl_p99_ms": 15.0}, {"itl_p99_ms": 0.0}])
    assert servable(r, Slo(20.0), GPUS) == (False, "itl_p99 25.0 ms > 20 ms")


def test_missing_metric_fails():
    r = rec(itl=None, inst=[{"tokens_per_sec": 1.0}])
    assert servable(r, Slo(20.0), GPUS) == (False, "itl_p99 SLO declared but not measured")
    r = rec(ttft=None, inst=[{}])
    assert servable(r, Slo(20.0, ttft_p99_ms=500.0), GPUS) == (False, "ttft_p99 SLO declared but not measured")


def test_acceptance_divides_itl():
    r = rec(itl=30.0, al=2.0)
    assert servable(r, Slo(20.0), GPUS) == (True, None)
    assert servable(rec(itl=50.0, al=2.0), Slo(20.0), GPUS) == (False, "itl_p99 25.0 ms > 20 ms")
    assert servable(rec(itl=30.0, al=0), Slo(20.0), GPUS)[0] is False  # non-positive -> 1.0


def test_ttft_and_e2e_worst_instance():
    slo = Slo(20.0, ttft_p99_ms=500.0, e2e_p99_ms=5000.0)
    r = rec(ttft=100.0, inst=[{"ttft_p99_ms": 100.0}, {"ttft_p99_ms": 600.0}])
    assert servable(r, slo, GPUS) == (False, "ttft_p99 600.0 ms > 500 ms")
    r = rec(e2e=1000.0, al=4.0, inst=[{"e2e_p99_ms": 6000.25}])  # no acceptance division for e2e
    assert servable(r, slo, GPUS) == (False, "e2e_p99 6000.2 ms > 5000 ms")
    assert servable(rec(), slo, GPUS) == (True, None)


def test_returncode_error_and_throughput_checks():
    assert servable(rec(rc=2, error=None), Slo(), GPUS) == (False, "rc=2")
    assert servable(rec(rc=-1, error="timeout"), Slo(), GPUS) == (False, "timeout")
    assert servable(rec(error="incomplete: 3 left"), Slo(), GPUS) == (False, "incomplete: 3 left")
    assert servable(rec(tps=0), Slo(), GPUS) == (False, "no throughput reported")
    assert servable(rec(tps=None), NO_SLO, GPUS) == (False, "no throughput reported")
    assert servable(rec(itl=999.0), NO_SLO, GPUS) == (True, None)


def test_rounded():
    assert rounded(rec(tps=323.555), GPUS) == (round(323.555 / 8, 2), 323.6)
    assert rounded(rec(tps=None), GPUS) == (None, None)
    assert rounded(rec(tps=0), GPUS) == (None, None)


# --- climb -------------------------------------------------------------------------------------

def test_climb_returns_probe_result_with_ladder_diagnostics(tmp_path):
    r = runner(tmp_path, lambda c: rec(tps=100.0 * c))
    p = climb(r, CFG, SC)
    assert isinstance(p, ProbeResult)
    assert r.ran == list(RUNGS)
    assert p.c_star is None and p.c_use == 8
    assert p.score == -(800.0 / GPUS) and p.tps_per_gpu == 800.0 / GPUS
    assert p.metrics_at_use["tokens_per_sec"] == 800.0
    d = p.diagnostics
    assert d["scoring"] == "ladder" and d["skipped"] == [] and d["best_rounded"] == 100.0
    assert d["rungs"][0] == {"c": 1, "servable": True, "why": None, "tps_per_gpu_rounded": 12.5,
                             "tps_rounded": 100.0}
    assert d["binding"] is None and d["ttft_bound"] is False and d["c_star_censored"] is False
    assert d["preemptions"] == 0 and d["kv_starved"] is False
    assert p.runs == len(p.keys) == 4


def test_climb_passes_seed_and_no_shed_guard():
    f = FakeRunner({c: rec() for c in RUNGS})
    climb(f, CFG, SC, seed=7)
    assert f.calls == [(c, 7, False) for c in RUNGS]


def test_error_rung_unservable_and_charged(tmp_path):
    sc = replace(SC, early_stop=False)
    r = runner(tmp_path, lambda c: rec(error="incomplete: 2 still queued") if c == 4 else rec(tps=100.0 * c), sc)
    p = climb(r, CFG, sc)
    rung = p.diagnostics["rungs"][2]
    assert rung["servable"] is False and rung["why"] == "incomplete: 2 still queued"
    assert p.runs == 4 and p.c_use == 8
    assert p.diagnostics["binding"] == "error"


def test_no_shed_guard_in_ladder(tmp_path):
    # 10 injected < requests_for(c) = 80 offered: the probe's shed guard fails it, the ladder does not
    r = runner(tmp_path, lambda c: rec(injected=10))
    p = climb(r, CFG, SC)
    assert all(g["servable"] for g in p.diagnostics["rungs"]) and p.score is not None
    pr = probe(r, CFG, replace(SC, scoring="probe", concurrency=None))
    assert pr.score is None


def test_shed_guard_off_leaves_cache_unchanged(tmp_path):
    r = runner(tmp_path, lambda c: rec(injected=10))
    off = r.run(CFG, 4, shed_guard=False)
    on = r.run(CFG, 4)
    assert off["error"] is None and on["error"].startswith("shed:")
    raw = (tmp_path / off["cache_key"][:2] / f"{off['cache_key']}.json").read_text()
    assert "shed:" not in raw


def test_early_stop_slo_first_miss(tmp_path):
    ok = {1: True, 2: False, 4: True, 8: True}
    r = runner(tmp_path, lambda c: rec(itl=10.0 if ok[c] else 30.0, tps=100.0 * c))
    p = climb(r, CFG, SC)
    assert r.ran == [1, 2]
    assert p.diagnostics["skipped"] == [4, 8]
    assert p.c_use == 1 and p.runs == 2
    assert p.diagnostics["binding"] == "itl"


def test_last_rung_never_stops(tmp_path):
    r = runner(tmp_path, lambda c: rec(itl=30.0 if c == 8 else 10.0, tps=100.0 * c))
    p = climb(r, CFG, SC)
    assert r.ran == list(RUNGS) and p.diagnostics["skipped"] == []
    assert p.c_use == 4


def test_early_stop_no_slo_fall(tmp_path):
    sc = replace(SC, slo=NO_SLO)
    tps = {1: 100.0, 2: 200.0, 4: 150.0, 8: 900.0}
    r = runner(tmp_path, lambda c: rec(itl=999.0, tps=tps[c]), sc)
    p = climb(r, CFG, sc)
    assert r.ran == [1, 2, 4] and p.diagnostics["skipped"] == [8]
    assert p.c_use == 2 and p.score == -(200.0 / GPUS)


def test_errored_rung_counts_zero_in_fall(tmp_path):
    sc = replace(SC, slo=NO_SLO)
    r = runner(tmp_path, lambda c: rec(error="boom") if c == 2 else rec(tps=100.0 * c), sc)
    p = climb(r, CFG, sc)
    assert r.ran == [1, 2] and p.diagnostics["skipped"] == [4, 8]
    assert p.c_use == 1


def test_no_slo_equal_throughput_does_not_stop(tmp_path):
    sc = replace(SC, slo=NO_SLO)
    r = runner(tmp_path, lambda c: rec(tps=800.0), sc)
    p = climb(r, CFG, sc)
    assert r.ran == list(RUNGS) and p.c_use == 1  # ties go to the lowest concurrency


def test_early_stop_off_runs_all(tmp_path):
    sc = replace(SC, early_stop=False)
    r = runner(tmp_path, lambda c: rec(itl=30.0 if c == 2 else 10.0, tps=100.0 * c), sc)
    p = climb(r, CFG, sc)
    assert r.ran == list(RUNGS) and p.diagnostics["skipped"] == [] and p.c_use == 8
    sc = replace(sc, slo=NO_SLO)
    r = runner(tmp_path / "b", lambda c: rec(tps={1: 400.0, 2: 100.0, 4: 300.0, 8: 200.0}[c]), sc)
    p = climb(r, CFG, sc)
    assert r.ran == list(RUNGS) and p.c_use == 1


def test_score_is_exact_best_rounded_kept(tmp_path):
    sc = replace(SC, concurrency=(1, 2))
    tps = {1: 12.345 * GPUS, 2: 12.341 * GPUS}
    r = runner(tmp_path, lambda c: rec(tps=tps[c]), sc)
    p = climb(r, CFG, sc)
    assert p.c_use == 1 and p.score == pytest.approx(-12.345, abs=1e-12)
    assert p.score == -(tps[1] / GPUS) and p.tps_per_gpu == tps[1] / GPUS
    assert p.diagnostics["best_rounded"] == 12.35


def test_nothing_servable_is_none(tmp_path):
    r = runner(tmp_path, lambda c: rec(itl=50.0))
    p = climb(r, CFG, SC)
    assert p.score is None and p.c_use is None and p.tps_per_gpu is None and p.c_star is None
    assert p.metrics_at_use == {} and p.diagnostics["best_rounded"] == 0.0
    assert p.runs == 1 and p.diagnostics["skipped"] == [2, 4, 8]


def test_preemptions_from_best_rung(tmp_path):
    def by_c(c):
        r = rec(tps=100.0 * c)
        r["metrics"]["preemption_count"] = 3 if c == 8 else 0
        return r
    p = climb(runner(tmp_path, by_c), CFG, SC)
    assert p.diagnostics["preemptions"] == 3 and p.diagnostics["kv_starved"] is True


def test_ttft_bound_at_first_miss(tmp_path):
    sc = replace(SC, slo=Slo(20.0, ttft_p99_ms=500.0))
    r = runner(tmp_path, lambda c: rec(ttft=900.0 if c >= 4 else 100.0, tps=100.0 * c), sc)
    p = climb(r, CFG, sc)
    assert p.diagnostics["binding"] == "ttft" and p.diagnostics["ttft_bound"] is True
    assert p.diagnostics["skipped"] == [8]


# --- early stop, written from the rule ----------------------------------------------------------

def test_early_stop_with_an_slo_stops_at_the_first_miss():
    # c=16 misses the ITL cap; c=32 would pass and be the best, but is never run
    recs = {8: rec(itl=10.0, tps=800.0), 16: rec(itl=21.0, tps=1600.0), 32: rec(itl=5.0, tps=9000.0)}
    sc = replace(SC, concurrency=(8, 16, 32))
    f = FakeRunner(recs)
    p = climb(f, CFG, sc)
    assert [c for c, _, _ in f.calls] == [8, 16]
    assert [g["servable"] for g in p.diagnostics["rungs"]] == [True, False]
    assert p.diagnostics["rungs"][1]["why"] == "itl_p99 21.0 ms > 20 ms"
    assert p.diagnostics["skipped"] == [32]
    assert p.c_use == 8 and p.score == -(800.0 / GPUS)


def test_early_stop_without_an_slo_stops_when_throughput_falls():
    # rounded tok/s/GPU 100, 150, 140 -> stop after c=4; c=8 would be higher but is never run
    recs = {1: rec(tps=800.0), 2: rec(tps=1200.0), 4: rec(tps=1120.0), 8: rec(tps=4000.0)}
    sc = replace(SC, slo=NO_SLO)
    f = FakeRunner(recs)
    p = climb(f, CFG, sc)
    assert [c for c, _, _ in f.calls] == [1, 2, 4]
    assert p.diagnostics["skipped"] == [8]
    assert [g["tps_per_gpu_rounded"] for g in p.diagnostics["rungs"]] == [100.0, 150.0, 140.0]
    assert p.c_use == 2 and p.score == -150.0 and p.diagnostics["best_rounded"] == 150.0
