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
    assert all(is_feasible(sp.knob_values(d), sp.scenario) for d in obj.calls)


def test_random_unbiased_samples_infeasible():
    sp = load_space(TINY)
    obj = SyntheticObjective(sp, lambda c: 0.0)
    RandomSearcher(sp, feasible_only=False, workers=1).search(obj, 64, 0)
    assert not all(is_feasible(sp.knob_values(d), sp.scenario) for d in obj.calls)


def test_random_stops_when_space_exhausted():
    sp = load_space(TINY)
    obj = SyntheticObjective(sp, lambda c: 0.0)
    t = RandomSearcher(sp, workers=1).search(obj, 10_000, 0)
    assert t.points[-1].runs < 10_000  # stalls on duplicates instead of looping


# --- P2a: applicability (requires) is structural, so every arm respects it ------------------------

def stray_labels(sp, d):
    """Names of knobs labelled in d although their `requires` is unmet."""
    names = sp.names
    out = []
    for i, k in enumerate(sp.knobs):
        if k.requires is not None and d[i] is not None:
            j = d[names.index(k.requires[0])]
            if j is None or sp.knobs[names.index(k.requires[0])].values[j] not in k.requires[1]:
                out.append(k.name)
    return out


def test_plain_bsa_and_unbiased_random_respect_applicability():
    from config_forage.search.bsa import BsaSearcher
    sp = load_space(TINY)
    for s in (BsaSearcher(sp, [], workers=1), RandomSearcher(sp, feasible_only=False, workers=1)):
        obj = SyntheticObjective(sp, lambda c: 0.0)
        t = s.search(obj, 120, 0)
        assert obj.calls and not [d for d in obj.calls if stray_labels(sp, d)]
        assert not [p for p in t.points if stray_labels(sp, p.decision)]
    assert BsaSearcher(sp, []).biasers == ()  # the attribute shows only the user's biasers


def test_applicability_biaser_and_canonical():
    from config_forage.search.bsa import BsaSearcher
    sp = load_space(TINY)
    b = sp.applicability_biaser()
    spec = sp.names.index("speculative")
    assert b.factor(5, (0, 0, 0, 0, 0, None)) is None and b.factor(5, (0, 0, 0, 0, 1, None)) is not None
    assert b.requires == {5: (spec, (1,))}
    assert sp.canonical((0, 0, 0, 0, 0, 1)) == (0, 0, 0, 0, 0, None)
    assert sp.canonical((0, 0, 0, 0, 1, 1)) == (0, 0, 0, 0, 1, 1)
    plain = type(sp)(sp.scenario, tuple(k for k in sp.knobs if k.requires is None))
    assert plain.applicability_biaser() is None
    assert BsaSearcher(plain, []).search(SyntheticObjective(plain, lambda c: 0.0), 20, 0).points
