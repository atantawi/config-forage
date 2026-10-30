"""The committed real-BLIS spaces load and have the sizes the study relies on (no BLIS needed)."""

from pathlib import Path

import math
from itertools import product

from config_forage.feasibility import TOPOLOGY, enumerate_feasible, is_feasible
from config_forage.space import KnobSpace, load_space

SPACES = Path(__file__).resolve().parent.parent / "spaces"
MID = SPACES / "glm53flash-h100-chatbot-16.yaml"
LARGE = SPACES / "glm53flash-h100-chatbot-16-large.yaml"


def count_feasible(sp: KnobSpace) -> int:
    """Feasible count without enumerating: walk the assignments of the knobs that matter for
    feasibility or gating (topology knobs and every `requires` parent), and multiply each by the
    sizes of the other knobs that apply under it (one level of gating: a parent is never gated)."""
    gated = [k for k in sp.knobs if k.requires]
    parents = {k.requires[0] for k in gated}
    assert not parents & {k.name for k in gated}, "one level of gating"
    walk = [k for k in sp.knobs if k.name in TOPOLOGY or k.name in parents]
    free = math.prod(len(k.values) for k in sp.knobs if k not in walk and not k.requires)
    n = 0
    for vs in product(*(k.values for k in walk)):
        a = dict(zip((k.name for k in walk), vs))
        if is_feasible(a, sp.scenario):
            n += math.prod(len(k.values) if a[k.requires[0]] in k.requires[1] else 1 for k in gated)
    return n * free


def test_count_feasible_matches_enumeration():
    sp = load_space(Path(__file__).parent / "data" / "tiny-space.yaml")
    assert count_feasible(sp) == sum(1 for _ in enumerate_feasible(sp))


def test_s_mid_size():
    sp = load_space(MID)
    assert sum(1 for _ in enumerate_feasible(sp)) == count_feasible(sp) == 3840  # <= 5k, all scoreable (min_footprint 8): full grid = ground truth
    assert sp.default_config()["num_instances"] == 2


def _with(sp: KnobSpace, **over) -> dict:
    """Each knob at its default, with overrides; gated knobs only where they apply."""
    c: dict = {}
    for k in sp.knobs:
        if k.requires is None or c.get(k.requires[0]) in k.requires[1]:
            c[k.name] = over.get(k.name, k.default)
    return c


def test_s_large_size_and_superset():
    mid, large = load_space(MID), load_space(LARGE)
    n = count_feasible(large)
    assert n >= 10**5, n
    assert n == 707_788_800
    for k in mid.knobs:  # S-mid is a sub-space of S-large, with the same defaults
        kl = large.knobs[large.names.index(k.name)]
        assert set(k.values) <= set(kl.values) and kl.default == k.default
    # ... and a knob only S-large has defaults to what S-mid runs without it: S-large's default config
    # translates to the same BLIS config as S-mid's (router weights included)
    from config_forage import router
    for fam in ("load-only", "inflight-load", "prefix-affinity"):
        cm = mid.config(mid.decision(_with(mid, endpoint_picker_config=fam)))
        cl = large.config(large.decision(_with(large, endpoint_picker_config=fam)))
        assert router.to_blis(cm)["epp_config"] == router.to_blis(cl)["epp_config"], fam
    assert mid.scenario == large.scenario
