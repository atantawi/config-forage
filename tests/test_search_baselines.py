from pathlib import Path

import pytest

from config_forage.feasibility import is_feasible
from config_forage.search.smac import SmacSearcher, to_configspace
from config_forage.search.tpe import TpeSearcher
from config_forage.space import load_space
from helpers import SyntheticObjective, synthetic_space

# pytest resets warning filters per test, over tpe.py's module-level filter; a mark outranks -W error.
pytestmark = pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")

TINY = Path(__file__).parent / "data" / "tiny-space.yaml"


def test_tpe_runs_and_respects_budget():
    sp = synthetic_space()
    t = TpeSearcher(sp).search(SyntheticObjective(sp), 40, 0)
    assert t.method == "tpe" and t.extra["stop"] == "budget"
    assert 40 <= t.points[-1].runs <= 41 and t.points[-1].best_score is not None


def test_tpe_prunes_infeasible_for_free():
    sp = load_space(TINY)
    obj = SyntheticObjective(sp, lambda c: float(c["max_num_seqs"]))
    t = TpeSearcher(sp).search(obj, 30, 0)
    assert all(is_feasible(sp.knob_values(d), sp.scenario) for d in obj.calls)
    assert t.extra["infeasible"] > 0 and len(t.points) == len(obj.calls)


def test_smac_never_proposes_infeasible():
    sp = load_space(TINY)
    obj = SyntheticObjective(sp, lambda c: float(c["max_num_seqs"]))
    t = SmacSearcher(sp).search(obj, 30, 0)
    assert t.method == "smac" and t.points[-1].runs <= 31
    assert obj.calls and all(is_feasible(sp.knob_values(d), sp.scenario) for d in obj.calls)


def test_failed_probe_is_worst_not_best():
    sp = synthetic_space()
    for s in (TpeSearcher(sp, n_startup=3), SmacSearcher(sp, n_init=3)):
        t = s.search(SyntheticObjective(sp, lambda c: None if c["a"] == 0 else -1.0), 15, 0)
        assert t.points[-1].best_score == -1.0


def test_to_configspace_conditions():
    sp = load_space(TINY)
    cs = to_configspace(sp)
    off = dict(cs.get_default_configuration())
    assert off["speculative"] == "False" and "num_speculative_tokens" not in off
    on = [dict(c) for c in cs.sample_configuration(200) if c["speculative"] == "True"]
    assert on and all("num_speculative_tokens" in c for c in on)
    assert all(is_feasible(sp.config(_decision(sp, c)), sp.scenario)
               for c in map(dict, cs.sample_configuration(200)))


def _decision(sp, c):
    return tuple(None if k.name not in c else tuple(map(str, k.values)).index(c[k.name]) for k in sp.knobs)


def test_baselines_deterministic_per_seed():
    sp = load_space(TINY)
    for mk in (lambda: TpeSearcher(sp), lambda: SmacSearcher(sp)):
        a, b = (mk().search(SyntheticObjective(sp, lambda c: float(c["max_num_seqs"] - c["tp"])), 25, 3)
                for _ in range(2))
        assert a.points == b.points and a.extra == b.extra


def test_tpe_records_pruned_positions():
    sp = load_space(TINY)
    obj = SyntheticObjective(sp, lambda c: float(c["max_num_seqs"]))
    t = TpeSearcher(sp).search(obj, 30, 0)
    pa, pp = t.extra["pruned_at"], t.extra["pruned_after"]
    assert len(pa) == len(pp) == t.extra["infeasible"] > 0
    assert pa == sorted(pa) and pp == sorted(pp) and all(0 <= k <= len(t.points) for k in pp)
    for q, k in zip(pa, pp):  # runs_used at a prune = runs of the last point recorded before it
        assert q == (t.points[k - 1].runs if k else 0)


def test_searcher_startup_on_s_large_is_fast():
    """Construct + the empty-space check (+ topology enumeration for incumbent) at budget 0."""
    from config_forage.feasibility import FeasibilityBiaser
    from config_forage.search.bsa import BsaSearcher
    from config_forage.search.coord import IncumbentSearcher
    from config_forage.search.random import RandomSearcher
    from helpers import Deadline
    sp = load_space(Path(__file__).resolve().parent.parent / "spaces" / "glm53flash-h100-chatbot-16-large.yaml")
    makers = [lambda: BsaSearcher(sp, [FeasibilityBiaser(sp)]), lambda: BsaSearcher(sp),
              lambda: RandomSearcher(sp), lambda: RandomSearcher(sp, feasible_only=False),
              lambda: TpeSearcher(sp), lambda: SmacSearcher(sp), lambda: IncumbentSearcher(sp)]
    with Deadline(15):
        for make in makers:
            t = make().search(SyntheticObjective(sp, lambda c: -1.0), 0, 0)
            assert t.points == []
