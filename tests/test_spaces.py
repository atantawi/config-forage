"""The committed real-BLIS spaces load and have the sizes the study relies on (no BLIS needed)."""

from pathlib import Path

from config_forage.feasibility import TOPOLOGY, enumerate_feasible
from config_forage.space import KnobSpace, load_space

SPACES = Path(__file__).resolve().parent.parent / "spaces"
MID = SPACES / "glm53flash-h100-chatbot-16.yaml"
LARGE = SPACES / "glm53flash-h100-chatbot-16-large.yaml"


def count_feasible(sp: KnobSpace) -> int:
    """Feasible count without enumerating: feasible topologies x product of the other knobs, where a
    knob gated by `requires` multiplies only its parent's matching values (one level of gating)."""
    topo = [k for k in sp.knobs if k.name in TOPOLOGY]
    rest = [k for k in sp.knobs if k.name not in TOPOLOGY]
    sub = KnobSpace(sp.scenario, tuple(topo))
    n = sum(1 for _ in enumerate_feasible(sub))
    gated = {k.requires[0]: k for k in rest if k.requires}
    assert len(gated) == sum(1 for k in rest if k.requires), "one gated knob per parent"
    for k in rest:
        if k.requires:
            assert k.requires[0] not in TOPOLOGY and k.name not in gated
            continue
        g = gated.get(k.name)
        n *= len(k.values) if g is None else sum(len(g.values) if v in g.requires[1] else 1 for v in k.values)
    return n


def test_count_feasible_matches_enumeration():
    sp = load_space(Path(__file__).parent / "data" / "tiny-space.yaml")
    assert count_feasible(sp) == sum(1 for _ in enumerate_feasible(sp))


def test_s_mid_size():
    sp = load_space(MID)
    assert sum(1 for _ in enumerate_feasible(sp)) == count_feasible(sp) == 4608  # <= 5k, all scoreable (min_footprint 8): full grid = ground truth
    assert sp.default_config()["num_instances"] == 2


def test_s_large_size_and_superset():
    mid, large = load_space(MID), load_space(LARGE)
    n = count_feasible(large)
    assert n >= 10**5, n
    assert n == 45_619_200
    for k in mid.knobs:  # S-mid is a sub-space of S-large, with the same defaults
        kl = large.knobs[large.names.index(k.name)]
        assert set(k.values) <= set(kl.values) and kl.default == k.default
    assert mid.scenario == large.scenario
