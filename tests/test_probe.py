import threading
from dataclasses import replace

from config_forage.blis.runner import CachedRunner
from config_forage.probe import Objective, ProbeResult, Slo, ladder, probe
from config_forage.space import Knob, KnobSpace, Scenario

SC = Scenario("m", "h100", "chatbot", 8, "0.29.0")
CFG = {"tp": 8, "num_instances": 1}
GPUS = 8


def fake(itl=lambda c: 5 + 0.5 * c, ttft=lambda c: 100.0, err=lambda c: None, gpus=GPUS, preempt=0):
    def fn(config, c, seed):
        return {"metrics": {"itl_p99_ms": itl(c), "ttft_p99_ms": ttft(c), "tokens_per_sec": 100.0 * c,
                            "preemption_count": preempt},
                "error": err(c), "returncode": 0, "deployment": {"gpus": gpus}}
    return fn


def runner(tmp_path, sc=SC, **kw):
    return CachedRunner(None, sc, tmp_path, run_fn=fake(**kw))


def space(sc=SC):
    return KnobSpace(sc, (Knob("tp", (8, 4)), Knob("max_num_seqs", (64, 128))))


def test_ladder_ported():
    assert ladder(8) == [1, 2, 4, 8, 16, 24, 32]
    assert ladder(64) == [1, 2, 4, 8, 16, 24, 32, 64, 128]


def test_probe_bisects_to_c_star(tmp_path):
    p = probe(runner(tmp_path), CFG, SC)
    assert isinstance(p, ProbeResult)
    assert p.c_star == 30 and p.c_use == 24
    assert p.score == -(2400 / GPUS) and p.tps_per_gpu == 2400 / GPUS
    assert p.metrics_at_use["tokens_per_sec"] == 2400
    assert p.diagnostics["binding"] == "itl" and not p.diagnostics["ttft_bound"]
    assert p.runs == len(p.keys) > 0


def test_probe_fails_at_c1_is_none(tmp_path):
    p = probe(runner(tmp_path, itl=lambda c: 50.0), CFG, SC)
    assert p.score is None and p.c_star is None and p.c_use is None and p.runs == 1


def test_probe_treats_run_error_as_fail(tmp_path):
    p = probe(runner(tmp_path, err=lambda c: "overloaded: still_queued=3" if c >= 20 else None), CFG, SC)
    assert p.c_star == 19 and p.c_use == 15 and p.diagnostics["binding"] == "error"


def test_probe_ladder_top_passes(tmp_path):
    p = probe(runner(tmp_path, itl=lambda c: 1.0), CFG, SC)
    assert p.c_star == ladder(8)[-1] and p.c_use == 25
    assert p.diagnostics["binding"] is None and p.diagnostics["ttft_bound"] is False


def test_probe_missing_gpus_is_none(tmp_path):
    assert probe(runner(tmp_path, gpus=0), CFG, SC).score is None


def test_probe_kv_starved(tmp_path):
    p = probe(runner(tmp_path, preempt=3), CFG, SC)
    assert p.diagnostics["kv_starved"] and p.diagnostics["preemptions"] == 3


def test_probe_c_use_fails_falls_back(tmp_path):
    p = probe(runner(tmp_path, itl=lambda c: 50.0 if c in (24, 18) else 5 + 0.5 * c), CFG, SC)
    # bisect 16..24 ⇒ C*=23, c_use=18 fails (non-monotone) ⇒ highest measured pass ≤ 18 is 16
    assert p.c_star == 23 and p.c_use == 16 and p.metrics_at_use["tokens_per_sec"] == 1600


def test_ttft_slo_applied(tmp_path):
    sc = replace(SC, slo=Slo(20.0, ttft_p99_ms=200.0))
    p = probe(runner(tmp_path, sc, ttft=lambda c: 10.0 * c), CFG, sc)
    assert p.c_star == 20 and p.c_use == 16
    assert p.diagnostics["binding"] == "ttft" and p.diagnostics["ttft_bound"] is True
    assert probe(runner(tmp_path / "b", ttft=lambda c: 10.0 * c), CFG, SC).c_star == 30  # no TTFT SLO


def test_probe_counts_unique_runs(tmp_path):
    obj = Objective(space(), runner(tmp_path))
    d = (0, 0)
    a = obj(d)
    n = obj.runs_used
    assert n == a.runs > 0
    obj(d)
    assert obj.runs_used == n and obj.evaluations == 2


def test_runs_used_counts_touched_keys_even_on_cache_hits(tmp_path):
    r = runner(tmp_path)
    Objective(space(), r)((0, 0))
    warm = r.unique_runs
    fresh = Objective(space(), r)
    p = fresh((0, 0))
    assert fresh.runs_used == len(p.keys) > 0 and r.unique_runs == warm


def test_runs_used_threads_union_keys(tmp_path):
    obj = Objective(space(), runner(tmp_path))
    out = {}
    ts = [threading.Thread(target=lambda d=d: out.__setitem__(d, obj(d))) for d in ((0, 0), (1, 1))]
    for t in ts: t.start()
    for t in ts: t.join()
    assert obj.runs_used == len(out[(0, 0)].keys | out[(1, 1)].keys) > out[(0, 0)].runs
