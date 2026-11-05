"""Optuna TPE baseline: conditional knobs suggested only where they apply; infeasible trials pruned for free."""

from __future__ import annotations

import warnings

import optuna

from config_forage.feasibility import feasible_topologies, is_feasible
from config_forage.search.base import Recorder, Trace
from config_forage.space import KnobSpace

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)

MAX_INFEASIBLE = 1000  # consecutive infeasible suggestions end the search
MAX_STALL = 80  # consecutive evaluated trials adding no unique run end the search
FAILED = 0.0  # a failed probe scores as zero throughput, the worst


class TpeSearcher:
    """One trial at a time via ask/tell. An infeasible suggestion is told PRUNED without calling the
    objective (0 runs, no trace point) and counted in trace.extra["infeasible"]; trace.extra["stop"]
    is "budget", "stall" or "infeasible".

    For the other cost convention (a pruned trial costs 1 run, as BLIS rejecting it would), each
    prune is logged: trace.extra["pruned_at"] holds runs_used at each pruned trial and
    trace.extra["pruned_after"] the number of trace points recorded before it (cf-analyze re-indexes
    the trace from these)."""

    def __init__(self, space: KnobSpace, multivariate: bool = True, n_startup: int = 10, name: str = "tpe"):
        self.space, self.multivariate, self.n_startup, self.name = space, multivariate, n_startup, name

    def search(self, objective, budget: int, seed: int) -> Trace:
        if not feasible_topologies(self.space):
            raise ValueError(f"no feasible decision in the space for scenario {self.space.scenario}")
        sampler = optuna.samplers.TPESampler(multivariate=self.multivariate, group=self.multivariate,
                                             n_startup_trials=self.n_startup, seed=seed)
        return self._loop(optuna.create_study(direction="minimize", sampler=sampler), objective, budget, seed)

    def _loop(self, study, objective, budget, seed) -> Trace:
        rec, labels = Recorder(self.space, objective), [tuple(map(str, k.values)) for k in self.space.knobs]
        infeasible = run_infeasible = stall = 0
        pruned_at: list[int] = []
        pruned_after: list[int] = []
        stop = "budget"
        while objective.runs_used < budget:
            if run_infeasible >= MAX_INFEASIBLE or stall >= MAX_STALL:
                stop = "infeasible" if run_infeasible >= MAX_INFEASIBLE else "stall"
                break
            trial, d, cfg = study.ask(), [], {}
            for k, opts in zip(self.space.knobs, labels):
                applies = k.requires is None or cfg.get(k.requires[0]) in k.requires[1]
                d.append(opts.index(trial.suggest_categorical(k.name, opts)) if applies else None)
                if applies:
                    cfg[k.name] = k.values[d[-1]]
            d = tuple(d)
            if not is_feasible(self.space.knob_values(d), self.space.scenario):
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                infeasible, run_infeasible = infeasible + 1, run_infeasible + 1
                pruned_at.append(objective.runs_used)
                pruned_after.append(len(rec.points))
                continue
            run_infeasible, before = 0, objective.runs_used
            r = rec.record(d, objective(d))
            study.tell(trial, FAILED if r.score is None else r.score)
            stall = stall + 1 if objective.runs_used == before else 0
        return Trace(rec.points, self.name, seed, {"infeasible": infeasible, "stop": stop,
                                                   "pruned_at": pruned_at, "pruned_after": pruned_after})


MAX_ASKS_PER_SLOT = 200  # asks per open batch slot before filling the slot in enumeration order


class TpeBatchSearcher:
    """Batched TPE for a ladder-scored space: Optuna TPESampler(multivariate, group, constant_liar)
    with Optuna's default startup. Each round asks for up to `batch` distinct new configs, evaluates
    them in parallel and tells the results in trial order, so `workers` changes wall time but not the
    trace. A round is sized so its worst case (every config climbing every rung) fits the remaining
    budget, so the budget is never exceeded.

    Asked picks that need no run are told at once: an infeasible one scores FAILED, one already
    scored gets its score, and one repeating a pick of the same round is told after the round runs.
    After MAX_ASKS_PER_SLOT asks per open slot the round is filled with unseen configs in
    enumeration order (not sampler trials; counted in trace.extra["fallback"]). Enumeration is lazy
    and resumed across rounds, so a large space is never walked unless the sampler keeps stalling.
    trace.extra["stop"] is "budget" or "exhausted" (every feasible config scored)."""

    def __init__(self, space: KnobSpace, batch: int = 8, workers: int | None = None, name: str = "tpe-batch"):
        self.space, self.batch, self.workers, self.name = space, batch, workers, name

    def search(self, objective, budget: int, seed: int) -> Trace:
        from config_forage.blis.runner import run_batch
        from config_forage.feasibility import enumerate_feasible

        sc = self.space.scenario
        if sc.scoring != "ladder":
            raise ValueError("tpe-batch sizes its batches by the ladder length: it needs scoring: ladder")
        if not feasible_topologies(self.space):
            raise ValueError(f"no feasible decision in the space for scenario {sc}")
        rungs = len(sc.concurrency)
        order = (self.space.canonical(d) for d in enumerate_feasible(self.space))  # lazy: fill-ins only
        with warnings.catch_warnings():  # group and constant_liar are flagged experimental
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True, seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        labels = [tuple(map(str, k.values)) for k in self.space.knobs]
        rec, keys, base = Recorder(self.space, objective), set(), objective.runs_used
        scores: dict = {}  # decision -> score told (FAILED for a failure)
        fallback, stop, exhausted = 0, "budget", False
        while True:
            n = min(self.batch, (budget - objective.runs_used) // rungs)
            if n <= 0:
                break
            picks, trials, repeats, asks = [], [], [], 0
            while len(picks) < n and asks < MAX_ASKS_PER_SLOT * n and not exhausted:
                trial, asks = study.ask(), asks + 1
                d = self._suggest(trial, labels)
                if not is_feasible(self.space.knob_values(d), sc):
                    study.tell(trial, FAILED)
                elif d in scores:
                    study.tell(trial, scores[d])
                elif d in picks:
                    repeats.append((trial, d))
                else:
                    picks.append(d)
                    trials.append(trial)
            while len(picks) < n and not exhausted:  # sampler stalled: fill in enumeration order
                d = next(order, None)
                if d is None:  # every feasible decision was met: the space is known from here on
                    exhausted = True
                    break
                if d not in scores and d not in picks:
                    picks.append(d)
                    trials.append(None)
                    fallback += 1
            if not picks:  # sampler and enumeration both found nothing unscored (so no repeats)
                stop = "exhausted"
                break
            results = run_batch(objective, picks, self.workers)
            after = objective.runs_used
            for trial, d, r in zip(trials, picks, results):
                scores[d] = FAILED if r.score is None else r.score
                if trial is not None:
                    study.tell(trial, scores[d])
                keys |= r.keys
                rec.record(d, r, min(base + len(keys), after))
            for trial, d in repeats:
                study.tell(trial, scores[d])
        return Trace(rec.points, self.name, seed, {"stop": stop, "batch": self.batch, "fallback": fallback})

    def _suggest(self, trial, labels):
        d, cfg = [], {}
        for k, opts in zip(self.space.knobs, labels):
            applies = k.requires is None or cfg.get(k.requires[0]) in k.requires[1]
            d.append(opts.index(trial.suggest_categorical(k.name, opts)) if applies else None)
            if applies:
                cfg[k.name] = k.values[d[-1]]
        return tuple(d)
