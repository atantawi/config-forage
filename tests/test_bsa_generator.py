import numpy as np

from config_forage.bsa.generator import CategoricalGenerator
from config_forage.bsa.space import Axis, DecisionSpace


def _space(*sizes):
    return DecisionSpace(tuple(Axis(f"a{i}", "", tuple(f"o{j}" for j in range(n))) for i, n in enumerate(sizes)))


SPACE6 = _space(4, 4, 3, 3, 2, 3)
SPACE5 = _space(5, 4, 5, 3, 2)
# axes 2, 3 apply only under option 3 of axis 1; axis 4 only when axis 0 took options 0-3
REQUIRES5 = {2: (1, (3,)), 3: (1, (3,)), 4: (0, (0, 1, 2, 3))}


def test_starts_uniform():
    g = CategoricalGenerator(SPACE6)
    assert all(np.allclose(p, 1 / len(p)) for p in g.P)
    assert np.isclose(g.entropy(), sum(np.log(n) for n in SPACE6.sizes))


def test_update_moves_toward_elites_and_respects_floor():
    g = CategoricalGenerator(SPACE6)
    ds = [(0, 0, 0, 0, 0, 0)] * 3 + [(1, 1, 1, 1, 1, 1)] * 7
    scores = [0.01] * 3 + [0.05] * 7  # lower is better
    elites = g.update(ds, scores)
    assert elites == [0, 1, 2]  # max(1, round(0.3 * 10)) = 3
    p0 = g.P[0]  # axis with 4 options
    q = 0.3 * 0.25 + 0.7 * 1.0
    assert np.isclose(p0[0], 0.02 + (1 - 4 * 0.02) * q)
    assert all(p.min() >= 0.02 - 1e-12 and np.isclose(p.sum(), 1) for p in g.P)


def test_failed_samples_are_not_elites():
    g = CategoricalGenerator(SPACE6)
    elites = g.update([(0,) * 6, (1,) * 6, (2, 2, 2, 2, 1, 2)], [None, 0.04, None])
    assert elites == [1]


def test_all_failed_leaves_P_unchanged():  # Review Focus 5
    g = CategoricalGenerator(SPACE6)
    before = g.snapshot()
    assert g.update([(0,) * 6] * 10, [None] * 10) == []
    assert g.snapshot() == before


def test_biaser_zero_excludes_option():
    class NoZero:
        def factor(self, axis, partial):
            f = np.ones(SPACE6.sizes[axis])
            f[0] = 0
            return f

    g = CategoricalGenerator(SPACE6)
    rng = np.random.default_rng(0)
    assert all(0 not in g.sample(rng, [NoZero()]) for _ in range(200))


def test_oracle_biaser_factor():
    from config_forage.bsa.biaser import OracleBiaser
    b = OracleBiaser(SPACE6, target=(3, 1, 2, 0, 0, 1), beta=2.0)
    assert np.allclose(b.factor(0, ()), [1, 1, 1, np.exp(2.0)])
    assert np.allclose(OracleBiaser(SPACE6, (3, 1, 2, 0, 0, 1), 0.0).factor(2, ()), 1)


def test_strong_oracle_bias_samples_target():
    from config_forage.bsa.biaser import OracleBiaser
    g = CategoricalGenerator(SPACE6)
    rng = np.random.default_rng(0)
    b = OracleBiaser(SPACE6, (3, 1, 2, 0, 0, 1), beta=30.0)
    assert all(g.sample(rng, [b]) == (3, 1, 2, 0, 0, 1) for _ in range(50))


def test_update_with_partial_labels():
    g = CategoricalGenerator(SPACE6)
    before = g.snapshot()
    g.update([(0, None, 0, 0, 0, 0)] * 3 + [(1,) * 6] * 7, [0.01] * 3 + [0.05] * 7)
    assert g.snapshot()[1] == before[1]  # axis 1: all elites unlabeled → unchanged
    assert g.P[0][0] > before[0][0]


def test_dead_end_sample_is_retried():
    from config_forage.bsa.generator import InfeasibleSample

    class DeadEnd:  # axis 1 has no feasible option when axis 0 took option 0
        def factor(self, axis, partial):
            f = np.ones(SPACE6.sizes[axis])
            if axis == 1 and partial[0] == 0:
                f[:] = 0
            return f

    g = CategoricalGenerator(SPACE6)
    rng = np.random.default_rng(0)
    assert all(g.sample(rng, [DeadEnd()])[0] != 0 for _ in range(200))

    class Never:
        def factor(self, axis, partial):
            return np.zeros(SPACE6.sizes[axis])

    import pytest
    with pytest.raises(InfeasibleSample):
        g.sample(rng, [Never()], max_tries=3)


def test_none_factor_leaves_axis_unlabelled_and_skips_later_biasers():
    asked = []

    class Skip2:
        def factor(self, axis, partial):
            return None if axis == 2 else np.ones(SPACE6.sizes[axis])

    class Spy:
        def factor(self, axis, partial):
            asked.append((axis, partial))
            return np.ones(SPACE6.sizes[axis])

    g = CategoricalGenerator(SPACE6)
    d = g.sample(np.random.default_rng(0), [Skip2(), Spy()])
    assert d[2] is None and all(x is not None for i, x in enumerate(d) if i != 2)
    assert 2 not in [a for a, _ in asked]
    assert asked[3][1][2] is None  # later axes see the unlabelled axis


def test_applicability_biaser_binpack2():
    import pytest

    from config_forage.bsa.biaser import ApplicabilityBiaser

    b = ApplicabilityBiaser(SPACE5, REQUIRES5)
    g = CategoricalGenerator(SPACE5)
    rng = np.random.default_rng(0)
    for _ in range(300):
        d = g.sample(rng, [b])
        target = d[1] == 3
        assert (d[2] is not None) == target and (d[3] is not None) == target
        assert (d[4] is not None) == (d[0] != 4)
        assert d[0] is not None and d[1] is not None
    with pytest.raises(ValueError):
        ApplicabilityBiaser(SPACE5, {1: (2, (0,))})  # depends on a later axis
    with pytest.raises(ValueError):
        ApplicabilityBiaser(SPACE5, {2: (1, (4,))})  # axis 1 has 4 options
    with pytest.raises(ValueError):
        ApplicabilityBiaser(SPACE5, {99: (0, (0,))})  # no axis 99


def test_update_ignores_axes_that_do_not_apply():
    g = CategoricalGenerator(SPACE5)
    before = g.snapshot()
    # elites use best fit, so R (axis 2) and the distance (axis 3) do not apply
    g.update([(0, 0, None, None, 0)] * 3 + [(1, 3, 1, 1, 1)] * 7, [0.01] * 3 + [0.05] * 7)
    assert g.snapshot()[2] == before[2] and g.snapshot()[3] == before[3]
    assert g.P[1][0] > before[1][0]


def test_rng_stays_in_step_whatever_the_biasers_answer():
    """Resume safety: a retry or a skipped axis must not shift later samples."""
    from config_forage.bsa.biaser import ApplicabilityBiaser

    class DeadEndOnce:  # zeroes the last axis on its first call: forces a retry
        calls = 0

        def factor(self, axis, partial):
            f = np.ones(SPACE5.sizes[axis])
            if axis == 4:
                DeadEndOnce.calls += 1
                if DeadEndOnce.calls == 1:
                    f[:] = 0
            return f

    g = CategoricalGenerator(SPACE5)
    skipped = 0
    for seed in range(10):
        for biasers in ([DeadEndOnce()], [ApplicabilityBiaser(SPACE5, REQUIRES5)]):
            DeadEndOnce.calls = 0
            plain, other = np.random.default_rng(seed), np.random.default_rng(seed)
            g.sample(plain)
            skipped += None in g.sample(other, biasers)
            assert [g.sample(plain) for _ in range(20)] == [g.sample(other) for _ in range(20)]
    assert skipped > 0  # some first samples did skip an axis


def test_sampling_matches_rng_choice():
    """Logs written before the uniform-per-axis sampler replay to the same decisions."""
    g = CategoricalGenerator(SPACE6)
    g.P = [np.random.default_rng(m).dirichlet(np.ones(n)) for m, n in enumerate(SPACE6.sizes)]
    a, b = np.random.default_rng(11), np.random.default_rng(11)

    def old(rng):
        return tuple(int(rng.choice(len(p), p=p / p.sum())) for p in g.P)

    assert [g.sample(a) for _ in range(500)] == [old(b) for _ in range(500)]


def test_bad_weights_raise_instead_of_picking_option_0():
    import pytest

    from config_forage.bsa.biaser import OracleBiaser

    s = DecisionSpace((Axis("a", "", ("x", "y")),))
    g = CategoricalGenerator(s)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        g.sample(np.random.default_rng(0), [OracleBiaser(s, (1,), beta=1000.0)])  # exp(1000) = inf

    class Big:  # with P = 0.5 and Two, each weight is 1.5e308 (finite) but the sum is inf
        def factor(self, axis, partial):
            return np.array([1.5e308, 1.5e308])

    class Two:
        def factor(self, axis, partial):
            return np.array([2.0, 2.0])

    class Negative:
        def factor(self, axis, partial):
            return np.array([-1.0, 2.0])

    for bs, msg in (([Big(), Two()], "overflow"), ([Negative()], "non-negative")):
        with np.errstate(over="ignore"), pytest.raises(ValueError, match=msg):
            g.sample(np.random.default_rng(0), bs)


def test_max_tries_must_be_positive():
    import pytest
    g = CategoricalGenerator(SPACE6)
    for n in (0, -1):
        with pytest.raises(ValueError, match="max_tries"):
            g.sample(np.random.default_rng(0), max_tries=n)


def test_hard_scheme_is_the_default_and_unchanged():
    g = CategoricalGenerator(SPACE6)
    assert g.scheme == "hard"
    g.update([(0,) * 6] * 3 + [(1,) * 6] * 7, [0.01] * 3 + [0.05] * 7)
    assert g.last_weights == [1 / 3] * 3 and np.isclose(g.ess, 3)


def test_bad_parameters_are_rejected():
    import pytest
    for kw in (dict(scheme="top"), dict(q=0), dict(q=1.5), dict(c=0), dict(min_ess=0.5)):
        with pytest.raises(ValueError):
            CategoricalGenerator(SPACE6, **kw)


def test_quantile_group_is_robust_to_an_outlier():
    """A cut at the median, not at the midpoint of the score range: one very bad sample (the
    c′ seed-0 shape) does not make nearly everything an elite."""
    scores = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 5.0]
    for scheme in ("boltzmann", "rank"):
        g = CategoricalGenerator(SPACE6, scheme=scheme)
        assert g.update([(0,) * 6] * 10, scores) == [0, 1, 2, 3, 4]
    assert CategoricalGenerator(SPACE6, scheme="boltzmann", q=0.2).update([(0,) * 6] * 10, scores) == [0, 1]


def test_ties_at_the_cut_are_all_elites():
    g = CategoricalGenerator(SPACE6, scheme="rank")
    elites = g.update([(0,) * 6] * 6, [0.1, 0.2, 0.2, 0.2, 0.2, 0.3])
    assert elites == [0, 1, 2, 3, 4]
    w = g.last_weights
    assert w[0] > w[1] and np.allclose(w[1:], w[1])  # tied scores share their weight


def test_boltzmann_weights_and_frequencies():
    g = CategoricalGenerator(SPACE6, scheme="boltzmann", c=1.0, min_ess=1)
    ds = [(0,) * 6, (1,) * 6, (2, 2, 2, 2, 1, 2), (3, 0, 0, 0, 0, 0)] + [(0,) * 6] * 4
    scores = [0.0, 0.1, 0.2, 0.4] + [9.0] * 4
    before = g.P[0].copy()
    assert g.update(ds, scores) == [0, 1, 2, 3]
    s = np.array(scores[:4])
    mad = np.median(np.abs(s - np.median(s)))  # 0.1
    w = np.exp(-s / mad)
    assert np.allclose(g.last_weights, w / w.sum())
    q = 0.3 * before + 0.7 * (w / w.sum())  # on axis 0, elite i chose option i
    assert np.allclose(g.P[0], 0.02 + (1 - 4 * 0.02) * q)


def test_boltzmann_equal_weights_when_the_group_has_no_spread():
    """Most repeats of a decision score the same (c′ seeds 2–4): MAD = 0 → equal weights."""
    g = CategoricalGenerator(SPACE6, scheme="boltzmann")
    g.update([(0,) * 6] * 10, [0.1] * 6 + [0.2] * 4)
    assert g.last_weights == [1 / 6] * 6


def test_boltzmann_ess_floor_stops_one_lucky_sample_taking_the_update():
    """The c′ seed-0 shape: a tight cluster and one sample far ahead. With tau = MAD of the
    cluster the leader would take all the weight; the floor widens tau instead."""
    scores = [-0.144] + [-0.035 - 0.001 * i for i in range(9)]  # negated gap-closed
    ds = [(0,) * 6] * 10
    loose = CategoricalGenerator(SPACE6, scheme="boltzmann", min_ess=1)
    loose.update(ds, scores)
    assert loose.ess < 1.01
    g = CategoricalGenerator(SPACE6, scheme="boltzmann")  # min_ess = 2
    g.update(ds, scores)
    assert 2 <= g.ess < 3 and np.argmax(g.last_weights) == 0


def test_weighted_update_with_partial_labels_renormalizes():
    g = CategoricalGenerator(SPACE6, scheme="rank")
    before = g.snapshot()
    g.update([(0, None, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1)] + [(2,) * 6] * 2, [0.01, 0.02, 0.5, 0.6])
    assert g.P[1][1] > before[1][1] and np.isclose(g.P[1].sum(), 1)  # axis 1: only elite 1 labelled
    assert np.isclose(g.P[1][1], 0.02 + (1 - 4 * 0.02) * (0.3 * 0.25 + 0.7))


def test_all_failed_resets_last_weights():
    g = CategoricalGenerator(SPACE6, scheme="boltzmann")
    g.update([(0,) * 6] * 4, [0.1, 0.2, 0.3, 0.4])
    assert g.update([(0,) * 6] * 4, [None] * 4) == [] and g.ess == 0.0


def test_boltzmann_weights_stay_finite_when_tau_underflows():
    """PR #8 review: tau = c * MAD can underflow to 0, and 0/0 made the weights and P NaN."""
    g = CategoricalGenerator(SPACE6, scheme="boltzmann", q=1, c=np.nextafter(0.0, 1.0))
    g.update([(0,) * 6, (1,) * 6, (2, 2, 2, 2, 1, 2)], [0.0, 0.5, 1.0])
    assert np.all(np.isfinite(g.last_weights)) and np.isclose(sum(g.last_weights), 1)
    assert all(np.all(np.isfinite(p)) and np.isclose(p.sum(), 1) for p in g.P)
    assert g.ess >= 2 - 1e-9  # the floor still widens tau from the underflowed start
    g = CategoricalGenerator(SPACE6, scheme="boltzmann", q=1, min_ess=1)
    g.update([(0,) * 6] * 3, [0.0, 1e308, 1.7e308])  # differences of huge scores overflow
    assert np.all(np.isfinite(g.last_weights)) and np.isclose(sum(g.last_weights), 1)


def test_weighted_schemes_reject_non_finite_scores():
    import pytest
    for scheme in ("boltzmann", "rank"):
        for bad in (np.nan, np.inf, -np.inf):
            g = CategoricalGenerator(SPACE6, scheme=scheme)
            before = g.snapshot()
            with pytest.raises(ValueError):
                g.update([(0,) * 6] * 3, [0.0, bad, 1.0])
            assert g.snapshot() == before


# --- review 2, finding 3: the eps floor must fit every axis ----------------------------------------

def test_eps_too_large_for_a_wide_axis_is_rejected():
    import pytest
    wide = DecisionSpace((*SPACE6.axes, Axis("wide", "", tuple(f"o{j}" for j in range(100)))))
    with pytest.raises(ValueError, match="wide") as e:
        CategoricalGenerator(wide, eps=0.02)
    msg = str(e.value)
    assert "100" in msg and "0.01" in msg  # the axis, its size and the largest eps allowed (< 1/100)
    for eps in (-0.01, 0.25, 0.3):  # SPACE6's widest axis has 4 options: eps * 4 must be < 1
        with pytest.raises(ValueError, match="eps"):
            CategoricalGenerator(SPACE6, eps=eps)


def test_eps_zero_is_allowed():
    g = CategoricalGenerator(SPACE6, eps=0)
    g.update([(0,) * 6] * 10, [1.0] * 10)
    assert all(p.min() >= 0 and np.isclose(p.sum(), 1) for p in g.P)


def test_alpha_and_rho_are_validated():
    import pytest
    for kw in ({"alpha": 0}, {"alpha": 1.5}, {"alpha": -0.1}, {"rho": 0}, {"rho": 1.01}):
        with pytest.raises(ValueError, match=next(iter(kw))):
            CategoricalGenerator(SPACE6, **kw)
    CategoricalGenerator(SPACE6, alpha=1, rho=1)  # the upper ends are allowed


def test_default_eps():
    from config_forage.bsa.generator import default_eps
    assert default_eps(SPACE6) == 0.02  # 0.5 / 4 > 0.02
    assert default_eps(_space(9, 2)) == 0.02  # S-large's widest axis
    assert default_eps(_space(100, 3)) == 0.005 and default_eps(_space(100), eps=0.001) == 0.001
