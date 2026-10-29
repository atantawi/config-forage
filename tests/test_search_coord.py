from config_forage.search.coord import CoordAscentSearcher, IncumbentSearcher
from tests.helpers import OPT_SCORE, SyntheticObjective, synthetic_space


def test_incumbent_misses_interaction():
    sp = synthetic_space()
    t = IncumbentSearcher(sp).search(SyntheticObjective(sp), 1000, 0)
    assert t.points[-1].best_score > OPT_SCORE
    assert t.extra["stop"] == "done"


def test_coord_ascent_reaches_separable_optimum():
    sp = synthetic_space()
    sep = lambda c: float((c["a"] - 4) ** 2 + (c["b"] - 1) ** 2 + (c["c"] - 3) ** 2 + (c["d"] - 2) ** 2)
    t = CoordAscentSearcher(sp).search(SyntheticObjective(sp, sep), 1000, 3)
    assert t.points[-1].best_score == 0.0
    assert t.extra["stop"] == "converged"


def test_coord_ascent_stuck_on_interaction():
    sp = synthetic_space()
    t = CoordAscentSearcher(sp).search(SyntheticObjective(sp), 1000, 0)
    assert t.points[-1].best_score == 0.0


def test_coord_stops_at_budget():
    sp = synthetic_space()
    for s in (CoordAscentSearcher(sp), IncumbentSearcher(sp)):
        o = SyntheticObjective(sp)
        t = s.search(o, 7, 0)
        assert o.runs_used == 7 and t.extra["stop"] == "budget"
