"""Optuna TPE baseline: conditional knobs suggested only where they apply; infeasible trials pruned for free."""

from __future__ import annotations

import warnings

import optuna

from config_forage.feasibility import enumerate_feasible, is_feasible
from config_forage.search.base import Recorder, Trace
from config_forage.space import KnobSpace

MAX_INFEASIBLE = 1000  # consecutive infeasible suggestions end the search
MAX_STALL = 80  # consecutive evaluated trials adding no unique run end the search
FAILED = 0.0  # a failed probe scores as zero throughput, the worst


class TpeSearcher:
    """One trial at a time via ask/tell. An infeasible suggestion is told PRUNED without calling the
    objective (0 runs, no trace point) and counted in trace.extra["infeasible"]; trace.extra["stop"]
    is "budget", "stall" or "infeasible"."""

    def __init__(self, space: KnobSpace, multivariate: bool = True, n_startup: int = 10, name: str = "tpe"):
        self.space, self.multivariate, self.n_startup, self.name = space, multivariate, n_startup, name

    def search(self, objective, budget: int, seed: int) -> Trace:
        if next(enumerate_feasible(self.space), None) is None:
            raise ValueError(f"no feasible decision in the space for scenario {self.space.scenario}")
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(multivariate=self.multivariate, group=self.multivariate,
                                                 n_startup_trials=self.n_startup, seed=seed)
            return self._loop(optuna.create_study(direction="minimize", sampler=sampler), objective, budget, seed)

    def _loop(self, study, objective, budget, seed) -> Trace:
        rec, labels = Recorder(self.space, objective), [tuple(map(str, k.values)) for k in self.space.knobs]
        infeasible = run_infeasible = stall = 0
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
            if not is_feasible(self.space.config(d), self.space.scenario):
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                infeasible, run_infeasible = infeasible + 1, run_infeasible + 1
                continue
            run_infeasible, before = 0, objective.runs_used
            r = rec.record(d, objective(d))
            study.tell(trial, FAILED if r.score is None else r.score)
            stall = stall + 1 if objective.runs_used == before else 0
        return Trace(rec.points, self.name, seed, {"infeasible": infeasible, "stop": stop})
