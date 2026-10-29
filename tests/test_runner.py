import threading
import time
from dataclasses import replace

from config_forage.blis.runner import CachedRunner, cache_key, run_batch, requests_for
from config_forage.space import Scenario

SC = Scenario("m", "h100", "chatbot", 8, "0.29.0")
CFG = {"tp": 8, "num_instances": 1}


def fake(calls, rc=0, delay=0.0):
    def fn(config, concurrency, seed):
        calls.append((dict(config), concurrency, seed))
        time.sleep(delay)
        return {"metrics": {"itl_p99_ms": 1.0}, "error": None if rc == 0 else "boom", "returncode": rc,
                "wall_s": 0.1, "deployment": {}}
    return fn


def test_cache_hit_does_not_call_blis(tmp_path):
    calls = []
    r = CachedRunner(None, SC, tmp_path, run_fn=fake(calls))
    a, b = r.run(CFG, 8), r.run(CFG, 8)
    assert len(calls) == 1 and r.unique_runs == 1 and r.cache_hits == 1
    assert a == b and a["cache_key"] == cache_key(CFG, 8, SC.seed, SC)


def test_cache_persists_across_instances(tmp_path):
    calls = []
    CachedRunner(None, SC, tmp_path, run_fn=fake(calls)).run(CFG, 8)
    r2 = CachedRunner(None, SC, tmp_path, run_fn=fake(calls))
    r2.run(CFG, 8)
    assert len(calls) == 1 and r2.unique_runs == 0 and r2.cache_hits == 1


def test_cache_key_includes_run_context():
    k = cache_key(CFG, 8, 0, SC)
    assert k != cache_key(CFG, 8, 1, SC)
    assert k != cache_key(CFG, 16, 0, SC)
    assert k != cache_key(CFG, 8, 0, replace(SC, workload="other"))
    assert cache_key(CFG, 8, 0, replace(SC, num_requests_base=100)) != k
    assert (cache_key(CFG, 8, 0, replace(SC, num_requests_base=20))
            == cache_key(CFG, 8, 0, replace(SC, num_requests_base=30)))  # 4c = 32 dominates
    assert cache_key({"num_instances": 1, "tp": 8}, 8, 0, SC) == k  # key order irrelevant


def test_seed_none_uses_scenario_seed(tmp_path):
    calls = []
    r = CachedRunner(None, replace(SC, seed=7), tmp_path, run_fn=fake(calls))
    r.run(CFG, 8)
    assert calls[0][2] == 7


def test_error_cached_timeout_not(tmp_path):
    calls = []
    r = CachedRunner(None, SC, tmp_path / "a", run_fn=fake(calls, rc=2))
    r.run(CFG, 8); r.run(CFG, 8)
    assert len(calls) == 1
    calls.clear()
    r = CachedRunner(None, SC, tmp_path / "b", run_fn=fake(calls, rc=-1))
    r.run(CFG, 8); r.run(CFG, 8)
    assert len(calls) == 2 and r.unique_runs == 2


def test_concurrent_same_key_runs_once(tmp_path):
    calls = []
    r = CachedRunner(None, SC, tmp_path, run_fn=fake(calls, delay=0.2))
    out = run_batch(lambda _: r.run(CFG, 8), list(range(6)), workers=6)
    assert len(calls) == 1 and r.unique_runs == 1 and r.cache_hits == 5
    assert len({o["cache_key"] for o in out}) == 1


def test_run_batch_preserves_order():
    assert run_batch(lambda x: x * 2, list(range(20)), workers=4) == [x * 2 for x in range(20)]
    assert run_batch(lambda x: x, []) == []


def test_requests_for():
    assert requests_for(8) == 80 and requests_for(100) == 400 and requests_for(8, 10) == 32


def test_non_json_values_in_record_are_cached(tmp_path):
    from pathlib import Path
    fn = lambda c, n, s: {"returncode": 0, "cmd": [Path("/x/blis")], "metrics": {}, "error": None}
    r = CachedRunner(None, SC, tmp_path, run_fn=fn)
    r.run(CFG, 8)
    assert CachedRunner(None, SC, tmp_path, run_fn=fn).run(CFG, 8)["cmd"] == ["/x/blis"]
