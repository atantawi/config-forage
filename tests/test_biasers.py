"""Diagnostic biaser and transfer prior. The BSA observe hook itself is covered by
test_search_bsa.py::test_observe_hook_called_per_generation (the brief's test_bsa_calls_observe)."""

import math

import numpy as np
import pytest
from helpers import SyntheticObjective, synthetic_space

from config_forage.biasers import DEFAULT_RULES, DiagnosticBiaser, Rule, TransferPrior
from config_forage.probe import ProbeResult
from config_forage.search.base import Trace
from config_forage.search.bsa import BsaSearcher
from config_forage.space import Knob, KnobSpace, Scenario

GMU = (0.8, 0.85, 0.9, 0.95)


def space() -> KnobSpace:
    sc = Scenario("dense", "h100", "chatbot", 8, "0.29.0")
    return KnobSpace(sc, (Knob("gpu_memory_utilization", GMU), Knob("cache_dtype", ("auto", "fp8")),
                          Knob("max_num_seqs", (64, 128, 256)), Knob("max_num_batched_tokens", (2048, 8192)),
                          Knob("block_size", (16,))))


def res(score, **diag) -> ProbeResult:
    d = {"binding": None, "ttft_bound": False, "kv_starved": False, "preemptions": 0, **diag}
    return ProbeResult(score, None, None, None, {}, d, 1)


def all_factors(b, sp):
    return [b.factor(m, (None,) * len(sp.knobs)) for m in range(len(sp.knobs))]


def test_diagnostic_neutral_before_observe():
    sp = space()
    for f, k in zip(all_factors(DiagnosticBiaser(sp, DEFAULT_RULES), sp), sp.knobs):
        assert np.array_equal(f, np.ones(len(k.values)))


def test_kv_starved_upweights_memory():
    sp = space()
    b = DiagnosticBiaser(sp, DEFAULT_RULES, beta=1.0)
    d = (0, 0, 0, 0, 0)
    b.observe([(d, res(-10.0 - i, kv_starved=True)) for i in range(10)])
    gmu, dtype, seqs, mnbt, bs = all_factors(b, sp)
    assert np.all(np.diff(gmu) > 0) and gmu[-1] == pytest.approx(math.e) and gmu[0] == pytest.approx(1 / math.e)
    assert dtype[1] == pytest.approx(math.e) and dtype[0] == 1.0
    assert np.all(np.diff(seqs) < 0)
    assert np.array_equal(mnbt, np.ones(2)) and np.array_equal(bs, np.ones(1))  # ttft not seen; single value


def test_signal_fraction_over_top_rho_elites():
    sp = space()
    b = DiagnosticBiaser(sp, [Rule("ttft_bound", "max_num_batched_tokens", "higher")], beta=2.0, rho=0.5)
    d = (0, 0, 0, 0, 0)
    # 4 successes -> 2 elites (scores -4, -3): one ttft-bound. Worse and failed results don't count.
    b.observe([(d, res(-4.0, ttft_bound=True)), (d, res(-3.0)), (d, res(-2.0, ttft_bound=True)),
               (d, res(-1.0, ttft_bound=True)), (d, res(None, ttft_bound=True))])
    f = b.factor(3, d)
    assert f == pytest.approx([math.exp(-1.0), math.exp(1.0)])  # beta * f_s = 2 * 0.5


def test_itl_binding_rule_and_multiplying_rules():
    sp = space()
    rules = [Rule("itl", "max_num_seqs", "lower"), Rule("kv_starved", "max_num_seqs", "lower")]
    b = DiagnosticBiaser(sp, rules, beta=1.0)
    d = (0, 0, 0, 0, 0)
    b.observe([(d, res(-1.0, binding="itl", kv_starved=True))])
    assert b.factor(2, d) == pytest.approx([math.e**2, 1.0, math.e**-2])


def test_failed_generation_keeps_factors():
    sp = space()
    b = DiagnosticBiaser(sp, DEFAULT_RULES)
    d = (0, 0, 0, 0, 0)
    b.observe([(d, res(-1.0, kv_starved=True))])
    before = b.factor(0, d)
    b.observe([(d, res(None))])
    assert np.array_equal(b.factor(0, d), before)
    b.observe([(d, res(-1.0))])  # successes without the signal: back to neutral
    assert np.array_equal(b.factor(0, d), np.ones(4))


def test_rules_on_missing_knob_or_value_ignored():
    sp = space()
    rules = [Rule("kv_starved", "no_such_knob", "higher"), Rule("kv_starved", "cache_dtype", "fp4")]
    b = DiagnosticBiaser(sp, rules)
    b.observe([((0, 0, 0, 0, 0), res(-1.0, kv_starved=True))])
    assert all(np.array_equal(f, np.ones(len(k.values))) for f, k in zip(all_factors(b, sp), sp.knobs))
    with pytest.raises(ValueError, match="signal"):
        DiagnosticBiaser(sp, [Rule("nonsense", "cache_dtype", "fp8")])


def test_diagnostic_never_zero():
    sp = space()
    b = DiagnosticBiaser(sp, DEFAULT_RULES * 50, beta=1e6)  # huge exponent: capped, no overflow
    b.observe([((0, 0, 0, 0, 0), res(-1.0, kv_starved=True, ttft_bound=True, binding="itl"))])
    for f in all_factors(b, sp):
        assert np.all(np.isfinite(f)) and np.all(f > 0)


def test_transfer_prior_tempering():
    snap = [[0.1, 0.6, 0.3], [0.5, 0.5]]
    assert all(np.array_equal(TransferPrior(snap, beta=0.0).factor(m, (None, None)), np.ones(len(p)))
               for m, p in enumerate(snap))
    assert all(np.allclose(TransferPrior(snap, beta=1.0).factor(m, (None, None)), p) for m, p in enumerate(snap))
    assert np.allclose(TransferPrior(snap).factor(0, (None, None)), np.sqrt(snap[0]))


def test_transfer_prior_validates_against_space():
    sp = synthetic_space()  # 4 axes of 6
    with pytest.raises(ValueError, match="sizes"):
        TransferPrior([[1 / 6] * 6] * 3, space=sp)
    with pytest.raises(ValueError, match="sizes"):
        TransferPrior([[1 / 6] * 6] * 3 + [[0.5, 0.5]], space=sp)
    with pytest.raises(ValueError, match="positive"):
        TransferPrior([[0.0, 1.0]])
    TransferPrior([[1 / 6] * 6] * 4, space=sp)


def test_transfer_prior_from_history(tmp_path):
    sp = synthetic_space()
    t = BsaSearcher(sp, workers=1).search(SyntheticObjective(sp), 48, 0)
    path = tmp_path / "trace.json"
    path.write_text(t.to_json())
    tp = TransferPrior.from_history(path, beta=1.0, space=sp)
    assert np.allclose(tp.factor(2, (None,) * 4), t.extra["history"][-1]["P"][2])
    with pytest.raises(ValueError, match="sizes"):
        TransferPrior.from_history(path, space=space())
    path.write_text(Trace([], "random", 0).to_json())
    with pytest.raises(ValueError, match="history"):
        TransferPrior.from_history(path)


def test_diagnostic_biaser_in_bsa():
    sp = synthetic_space()

    class KvStarved(SyntheticObjective):
        def __call__(self, d):
            r = super().__call__(d)
            return ProbeResult(r.score, None, None, None, {}, {"kv_starved": True}, r.runs, r.keys)

    b = DiagnosticBiaser(sp, [Rule("kv_starved", "c", "higher")])
    BsaSearcher(sp, biasers=[b], workers=1).search(KvStarved(sp), 32, 0)
    assert np.all(np.diff(b.factor(2, (None,) * 4)) > 0) and b.fractions["kv_starved"] == 1.0


def test_active_rules():
    from dataclasses import replace

    from config_forage.space import Slo
    sp = space()  # gmu 4 values, cache_dtype 2, max_num_seqs 3, mnbt 2, block_size 1
    rules = (*DEFAULT_RULES, Rule("itl", "block_size", "higher"), Rule("kv_starved", "cache_dtype", "fp16"),
             Rule("itl", "absent_knob", "lower"))
    act = DiagnosticBiaser(sp, rules).active_rules()
    assert act == [r for r in DEFAULT_RULES if r.signal != "ttft_bound"]  # no TTFT SLO: never ttft-bound
    sp_t = KnobSpace(replace(sp.scenario, slo=Slo(20.0, ttft_p99_ms=500.0)), sp.knobs)
    assert DiagnosticBiaser(sp_t, rules).active_rules() == list(DEFAULT_RULES)
    pinned = KnobSpace(sp.scenario, tuple(Knob(k.name, k.values[:1]) for k in sp.knobs))
    assert DiagnosticBiaser(pinned, DEFAULT_RULES).active_rules() == []
