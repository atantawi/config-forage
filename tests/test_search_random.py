from pathlib import Path

from config_forage.feasibility import is_feasible
from config_forage.search.random import RandomSearcher
from config_forage.space import load_space
from helpers import SyntheticObjective

TINY = Path(__file__).parent / "data" / "tiny-space.yaml"


def test_random_feasible_only():
    sp = load_space(TINY)
    obj = SyntheticObjective(sp, lambda c: float(c["tp"]))
    t = RandomSearcher(sp, workers=1).search(obj, 40, 0)
    assert t.method == "random" and len(t.points) >= 40
    assert all(is_feasible(sp.config(d), sp.scenario) for d in obj.calls)


def test_random_unbiased_samples_infeasible():
    sp = load_space(TINY)
    obj = SyntheticObjective(sp, lambda c: 0.0)
    RandomSearcher(sp, feasible_only=False, workers=1).search(obj, 64, 0)
    assert not all(is_feasible(sp.config(d), sp.scenario) for d in obj.calls)


def test_random_stops_when_space_exhausted():
    sp = load_space(TINY)
    obj = SyntheticObjective(sp, lambda c: 0.0)
    t = RandomSearcher(sp, workers=1).search(obj, 10_000, 0)
    assert t.points[-1].runs < 10_000  # stalls on duplicates instead of looping
