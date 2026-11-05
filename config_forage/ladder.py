"""Ladder scoring: run a fixed concurrency ladder and score the best servable rung.

A rung is servable when its run succeeded, reported throughput, and every declared p99 cap holds on
the worst of the aggregate and each instance (ITL divided by the speculative acceptance length).
Score = -(tokens_per_sec / gpus) of the servable rung with the highest exact tok/s/GPU (ties go to
the lower concurrency); None when no rung is servable. Lower is better, as for the C* probe.

Early stop (scenario.early_stop, never after the last rung): with an SLO declared, stop at the first
unservable rung; with none, stop when a rung's rounded tok/s/GPU (0 for a failed rung) falls below
the previous rung's. The rungs not run are listed in diagnostics["skipped"].

The ladder reads each record without the runner's shed guard (runner.run(..., shed_guard=False))."""

from __future__ import annotations

from config_forage.probe import ProbeResult
from config_forage.space import Scenario, Slo

__all__ = ["climb", "rounded", "servable", "worst_metric"]


def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def worst_metric(rec: dict, key: str) -> float | None:
    """Max of `key` over the aggregate metrics and every instance's; None/non-numeric skipped."""
    vals = [(rec.get("metrics") or {}).get(key)]
    vals += [(m or {}).get(key) for m in rec.get("metrics_all_instances") or []]
    nums = [x for x in map(_num, vals) if x is not None]
    return max(nums) if nums else None


def _acceptance(rec: dict) -> float:
    al = _num(((rec.get("deployment") or {}).get("acceptance") or {}).get("acceptance_length"))
    return al if al is not None and al > 0 else 1.0


def _failed_run(rec: dict) -> str | None:
    """The reason a run itself failed (checks 1-3 of servable), else None."""
    rc = rec.get("returncode")
    if rc != 0:
        return rec.get("error") or f"rc={rc}"
    if rec.get("error"):
        return rec["error"]
    if not _num((rec.get("metrics") or {}).get("tokens_per_sec")):
        return "no throughput reported"
    return None


def servable(rec: dict, slo: Slo, gpus: int | None) -> tuple[bool, str | None]:
    """(True, None), or (False, why) for the first failing check. `gpus` is unused here."""
    why = _failed_run(rec)
    if why is not None:
        return False, why
    if slo.itl_p99_ms is not None:
        cap, worst = slo.itl_p99_ms, worst_metric(rec, "itl_p99_ms")
        if worst is None:
            return False, "itl_p99 SLO declared but not measured"
        v = worst / _acceptance(rec)
        if v > cap:
            return False, f"itl_p99 {v:.1f} ms > {cap:g} ms"
    for name, cap in (("ttft_p99", slo.ttft_p99_ms), ("e2e_p99", slo.e2e_p99_ms)):
        if cap is None:
            continue
        worst = worst_metric(rec, f"{name}_ms")
        if worst is None:
            return False, f"{name} SLO declared but not measured"
        if worst > cap:
            return False, f"{name} {worst:.1f} ms > {cap:g} ms"
    return True, None


def rounded(rec: dict, gpus: int) -> tuple[float | None, float | None]:
    """(tok/s/GPU to 2 dp, tok/s to 1 dp); (None, None) without throughput or GPUs."""
    tps = _num((rec.get("metrics") or {}).get("tokens_per_sec"))
    if not tps or not gpus:
        return None, None
    return round(tps / gpus, 2), round(tps, 1)


def _slo_declared(slo: Slo) -> bool:
    return any(v is not None for v in (slo.itl_p99_ms, slo.ttft_p99_ms, slo.e2e_p99_ms))


def climb(runner, config: dict, scenario: Scenario, seed: int | None = None) -> ProbeResult:
    slo, cs = scenario.slo, list(scenario.concurrency or ())
    has_slo = _slo_declared(slo)
    recs: dict[int, dict] = {}
    rungs: list[dict] = []
    skipped: list[int] = []
    best = None  # (exact tps/GPU, c)
    first_miss = None
    prev_r2 = None
    for i, c in enumerate(cs):
        rec = recs[c] = runner.run(config, c, seed, shed_guard=False)
        gpus = (rec.get("deployment") or {}).get("gpus")
        ok, why = servable(rec, slo, gpus)
        r2, r1 = rounded(rec, gpus)
        rungs.append({"c": c, "servable": ok, "why": why, "tps_per_gpu_rounded": r2, "tps_rounded": r1})
        if ok and gpus:
            exact = float(rec["metrics"]["tokens_per_sec"]) / gpus
            if best is None or exact > best[0]:
                best = (exact, c)
        elif first_miss is None and not ok:
            first_miss = c
        r2 = (r2 or 0.0) if ok else 0.0  # a failed rung counts as 0 in the fall check
        fell = prev_r2 is not None and r2 < prev_r2
        if scenario.early_stop and i < len(cs) - 1 and (not ok if has_slo else fell):
            skipped = cs[i + 1:]
            break
        prev_r2 = r2

    best_rounded = max((g["tps_per_gpu_rounded"] for g in rungs
                        if g["servable"] and g["tps_per_gpu_rounded"] is not None), default=0.0)
    diag = {"scoring": "ladder", "rungs": rungs, "skipped": skipped, "best_rounded": best_rounded,
            "preemptions": 0, "kv_starved": False, "ttft_bound": False, "binding": None,
            "c_star_censored": False}
    if best is not None:
        pre = int((recs[best[1]].get("metrics") or {}).get("preemption_count") or 0)
        diag["preemptions"], diag["kv_starved"] = pre, pre > 0
    if first_miss is not None:
        f = recs[first_miss]
        why = next(g["why"] for g in rungs if g["c"] == first_miss)
        if _failed_run(f) is not None:
            diag["binding"] = "error"
        else:
            itl_bound = why.startswith("itl_p99")
            diag["binding"] = "itl" if itl_bound else "ttft"  # as the probe: e2e counts as "ttft"
            diag["ttft_bound"] = slo.ttft_p99_ms is not None and not itl_bound

    keys = frozenset(r["cache_key"] for r in recs.values())
    if best is None:
        return ProbeResult(None, None, None, None, {}, diag, len(keys), keys)
    exact, c_use = best
    return ProbeResult(-exact, None, c_use, exact, recs[c_use].get("metrics") or {}, diag, len(keys), keys)
