"""tpe-batch: batched constant-liar TPE, each batch sized so its worst case fits the budget."""

from dataclasses import replace

import pytest

from config_forage.bench.run import ARMS
from config_forage.search.tpe import TpeBatchSearcher
from config_forage.space import Knob, KnobSpace
from helpers import SyntheticObjective, synthetic_space

LADDER = (16, 64, 256, 1024)  # L = 4 rungs: a config's worst case is 4 runs


def ladder_space(knobs=None) -> KnobSpace:
    sp = synthetic_space()
    sc = replace(sp.scenario, scoring="ladder", concurrency=LADDER, headroom=1.0)
    return KnobSpace(sc, knobs or sp.knobs)


class VariableRuns(SyntheticObjective):
    """A new decision costs 1..4 runs (an early-stopped climb), fixed per decision; a repeat costs 0."""

    def __init__(self, space):
        super().__init__(space)
        self._cost: dict = {}

    def __call__(self, d):
        cost = 1 + sum(i or 0 for i in d) % 4
        r = super().__call__(d)
        with self._lock:
            self._cost.setdefault(d, cost)
        return replace(r, runs=cost, keys=frozenset(f"{d}:{i}" for i in range(cost)))

    @property
    def runs_used(self):
        with self._lock:
            return sum(self._cost.values())


@pytest.mark.parametrize("obj_cls", [SyntheticObjective, VariableRuns])
def test_never_exceeds_budget(obj_cls):
    sp = ladder_space()
    obj = obj_cls(sp, runs_per_eval=4) if obj_cls is SyntheticObjective else obj_cls(sp)
    t = TpeBatchSearcher(sp, workers=1).search(obj, 250, seed=0)
    assert obj.runs_used <= 250
    assert max(p.runs for p in t.points) > 200
    assert t.extra["stop"] == "budget"


def test_ends_when_space_exhausted():
    sp = ladder_space((Knob("a", tuple(range(6))),))
    obj = SyntheticObjective(sp, score_fn=lambda c: float(c["a"]), runs_per_eval=4)
    t = TpeBatchSearcher(sp, workers=1).search(obj, 1000, seed=0)
    assert t.extra["stop"] == "exhausted"
    assert sorted(obj.calls) == [(i,) for i in range(6)]  # each config once, none twice


def test_deterministic():
    sp = ladder_space()
    runs = [TpeBatchSearcher(sp, workers=w).search(SyntheticObjective(sp, runs_per_eval=2), 120, seed=3)
            for w in (1, 4)]
    assert [p.decision for p in runs[0].points] == [p.decision for p in runs[1].points]
    assert [p.runs for p in runs[0].points] == [p.runs for p in runs[1].points]


def test_probe_space_refused():
    sp = synthetic_space()
    with pytest.raises(ValueError, match="ladder"):
        TpeBatchSearcher(sp).search(SyntheticObjective(sp), 50, seed=0)


def test_batches_of_eight_then_smaller():
    sp = ladder_space()
    t = TpeBatchSearcher(sp, workers=1).search(SyntheticObjective(sp, runs_per_eval=4), 100, seed=0)
    # 100 // 4 = 25 configs at most: three batches of 8, then one of 1
    assert len(t.points) == 25 and t.extra["batch"] == 8


def test_finds_good_configs():
    sp = ladder_space()
    t = TpeBatchSearcher(sp, workers=1).search(SyntheticObjective(sp, runs_per_eval=1), 300, seed=0)
    assert t.points[-1].best_score <= -2.0  # random's median at 300 of 1296 is about 0


def test_arm_registered():
    s = ARMS["tpe-batch"](ladder_space(), {})
    assert isinstance(s, TpeBatchSearcher) and s.name == "tpe-batch"
