import json
import statistics

import numpy as np
import pytest

from config_forage.search.base import Trace, TracePoint
from config_forage.search.bsa import BsaSearcher
from config_forage.search.random import RandomSearcher
from helpers import OPT_SCORE, OPTIMUM, SyntheticObjective, empty_space, synthetic_score, synthetic_space


def run(searcher, budget, seed, **kw):
    return searcher.search(SyntheticObjective(searcher.space, **kw), budget, seed)


def test_bsa_finds_synthetic_optimum():
    sp = synthetic_space()
    found = 0
    for seed in range(5):
        t = run(BsaSearcher(sp, workers=1), 300, seed)
        found += t.points[-1].best_score == OPT_SCORE and t.points[-1].best_config == {**OPTIMUM, "num_instances": 8}
    assert found >= 4


def test_bsa_beats_random_median():
    sp = synthetic_space()
    bsa = [run(BsaSearcher(sp, workers=1), 150, s).value_at(150) for s in range(10)]
    rnd = [run(RandomSearcher(sp, workers=1), 150, s).value_at(150) for s in range(10)]
    assert statistics.median(bsa) < statistics.median(rnd)


def test_trace_monotone():
    sp = synthetic_space()
    t = run(BsaSearcher(sp, workers=4), 200, 0)
    best = [p.best_score for p in t.points]
    assert all(b is not None for b in best)
    assert all(y <= x for x, y in zip(best, best[1:]))
    assert [p.evaluated for p in t.points] == list(range(1, len(t.points) + 1))
    assert all(y >= x for x, y in zip([p.runs for p in t.points], [p.runs for p in t.points][1:]))


def test_trace_none_until_first_success():
    sp = synthetic_space()
    calls = []

    def score(cfg):
        calls.append(cfg)
        return None if len(calls) <= 3 else float(cfg["a"])

    t = BsaSearcher(sp, workers=1).search(SyntheticObjective(sp, score), 20, 0)
    assert [p.best_score for p in t.points[:3]] == [None] * 3 and t.points[0].best_config is None
    assert t.points[3].best_score is not None


def test_budget_overshoot_recorded():
    sp = synthetic_space()
    obj = SyntheticObjective(sp, runs_per_eval=3)
    t = BsaSearcher(sp, batch=16, workers=1).search(obj, 50, 0)
    assert obj.evaluations == 32  # 48 < 50 launches a second batch; 96 >= 50 stops
    assert t.points[-1].runs == obj.runs_used == 96
    assert t.points[15].runs == 48 and t.points[16].runs == 51
    assert t.value_at(50) == t.points[15].best_score


def test_bsa_rejects_empty_feasible_space():
    sp = empty_space()
    obj = SyntheticObjective(sp, lambda c: 0.0)
    with pytest.raises(ValueError, match="no feasible"):
        BsaSearcher(sp).search(obj, 10, 0)
    assert obj.evaluations == 0


def test_bsa_history_and_json_roundtrip():
    sp = synthetic_space()
    s = BsaSearcher(sp, workers=1)
    t = run(s, 48, 1)
    h = t.extra["history"]
    assert h is s.history and [g["gen"] for g in h] == [0, 1, 2]
    assert [g["runs"] for g in h] == [16, 32, 48]
    assert set(h[0]) == {"gen", "runs", "entropy", "P", "ess"} and len(h[0]["P"]) == 4
    assert h[-1]["entropy"] < h[0]["entropy"]
    t2 = Trace.from_json(t.to_json())
    assert t2 == t and isinstance(t2.points[0], TracePoint)
    json.loads(t.to_json())


def test_bsa_resamples_duplicates():
    sp = synthetic_space()
    obj = SyntheticObjective(sp)
    BsaSearcher(sp, workers=1).search(obj, 64, 0)
    assert len(set(obj.calls)) == len(obj.calls)  # 1296 combos: 10 resamples always find a fresh one early on


def test_observe_hook_called_per_generation():
    sp = synthetic_space()

    class Spy:
        def __init__(self):
            self.seen = []

        def factor(self, axis, partial):
            return np.ones(6)

        def observe(self, pairs):
            self.seen.append(pairs)

    spy = Spy()
    obj = SyntheticObjective(sp)
    BsaSearcher(sp, biasers=[spy], workers=1).search(obj, 32, 0)
    assert len(spy.seen) == 2 and all(len(g) == 16 for g in spy.seen)
    d, r = spy.seen[0][0]
    assert r.score == obj.score_fn(sp.knob_values(d))


def test_value_at():
    t = Trace([TracePoint(1, None, None, 1), TracePoint(3, 5.0, {"a": 1}, 2), TracePoint(5, 2.0, {"a": 2}, 3)], "x", 0)
    assert t.value_at(0) is None and t.value_at(1) is None
    assert t.value_at(4) == 5.0 and t.value_at(99) == 2.0


def test_synthetic_optimum_unique_and_coordinatewise_trap():
    from itertools import product

    from helpers import synthetic_score
    sp = synthetic_space()
    scores = sorted((synthetic_score(sp.knob_values(d)), d) for d in product(range(6), repeat=4))
    assert scores[0][0] == OPT_SCORE < scores[1][0] and sp.config(scores[0][1]) == {**OPTIMUM, "num_instances": 8}
    trap = {"a": 0, "b": 0, "c": 3, "d": 2}
    for k in "ab":
        assert all(synthetic_score({**trap, k: v}) > 0 for v in range(1, 6))


def test_seeded_trace_independent_of_thread_timing():
    import random
    import time

    sp = synthetic_space()

    class Jittery(SyntheticObjective):
        def __call__(self, d):
            time.sleep(random.uniform(0, 0.002))
            return super().__call__(d)

    a, b = (BsaSearcher(sp, workers=4).search(Jittery(sp), 80, 3) for _ in range(2))
    assert a.points == b.points and a.extra["history"] == b.extra["history"]


def test_trace_points_record_each_evaluation_and_roundtrip():
    sp = synthetic_space()
    t = BsaSearcher(sp, workers=1).search(SyntheticObjective(sp), 20, 0)
    for p in t.points:
        assert isinstance(p.decision, list) and p.score == synthetic_score(sp.knob_values(tuple(p.decision)))
    assert all(p.best_score == min(q.score for q in t.points[:i + 1]) for i, p in enumerate(t.points))
    assert Trace.from_json(t.to_json()).points == t.points


def test_trace_from_old_json_lacking_decision_and_score():
    import json
    old = {"points": [{"runs": 1, "best_score": -1.0, "best_config": {"a": 1}, "evaluated": 1}],
           "method": "x", "seed": 0, "extra": {}}
    (p,) = Trace.from_json(json.dumps(old)).points
    assert p.decision is None and p.score is None and p.best_score == -1.0


# --- review 2, finding 3: wide axes work with the default eps -----------------------------------------

def _wide_space():
    from config_forage.space import Knob, KnobSpace, Scenario
    sc = Scenario("synthetic", "h100", "chatbot", 8, "0.29.0")
    return KnobSpace(sc, (Knob("a", tuple(range(6))), Knob("w", tuple(range(100)))))


def test_bsa_on_a_100_option_axis_keeps_P_a_distribution():
    sp = _wide_space()
    s = BsaSearcher(sp, batch=16)
    t = s.search(SyntheticObjective(sp, lambda c: float(abs(c["w"] - 37) + c["a"])), 200, 0)
    assert t.points and len(s.history) > 3
    for h in s.history:
        for p in h["P"]:
            assert min(p) >= 0 and abs(sum(p) - 1) < 1e-9
    assert s.eps == 0.005  # default_eps: 0.5 / 100
    assert all(min(h["P"][1]) >= 0.005 - 1e-12 for h in s.history)


def test_random_on_a_100_option_axis():
    sp = _wide_space()
    t = RandomSearcher(sp, feasible_only=False).search(SyntheticObjective(sp, lambda c: 0.0), 50, 0)
    assert t.points


def test_default_eps_is_unchanged_on_s_mid_and_s_large():
    """Both shipped spaces keep eps = 0.02 (max axis 5 and 6, and 0.5 / 6 > 0.02), so BSA is
    bit-identical to the explicit eps = 0.02 there."""
    from pathlib import Path

    from config_forage.bsa.generator import default_eps
    from config_forage.space import load_space
    spaces = Path(__file__).resolve().parent.parent / "spaces"
    for f, widest in (("glm53flash-h100-chatbot-16.yaml", 5), ("glm53flash-h100-chatbot-16-large.yaml", 6)):
        sp = load_space(spaces / f)
        assert max(sp.decision_space.sizes) == widest and default_eps(sp.decision_space) == 0.02
    sp = synthetic_space()
    a = BsaSearcher(sp).search(SyntheticObjective(sp), 120, 3)
    b = BsaSearcher(sp, eps=0.02).search(SyntheticObjective(sp), 120, 3)
    assert a.to_json() == b.to_json()
