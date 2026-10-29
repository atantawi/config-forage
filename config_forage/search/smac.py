"""SMAC3 baseline: categorical hyperparameters, conditions from `requires`, forbiddens for infeasible topologies."""

from __future__ import annotations

import logging
import tempfile
from itertools import product
from pathlib import Path

from ConfigSpace import (Categorical, ConfigurationSpace, EqualsCondition, ForbiddenAndConjunction,
                         ForbiddenEqualsClause, InCondition)
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue
from smac.runhistory.enumerations import StatusType

from config_forage.feasibility import TOPOLOGY, enumerate_feasible, is_feasible
from config_forage.search.base import Recorder, Trace
from config_forage.space import KnobSpace

logging.getLogger("smac").setLevel(logging.ERROR)

MAX_STALL = 80  # consecutive evaluated trials adding no unique run end the search
FAILED = 0.0  # a failed probe costs zero throughput, the worst


def to_configspace(space: KnobSpace) -> ConfigurationSpace:
    """Each knob categorical over its option labels (str of the value), default at the knob default;
    a conditional knob is active only where its parent takes a required value; every infeasible
    topology combination is forbidden (feasibility depends only on the topology knobs)."""
    cs = ConfigurationSpace()
    hps = {k.name: Categorical(k.name, [str(v) for v in k.values], default=str(k.default)) for k in space.knobs}
    cs.add(list(hps.values()))
    for k in space.knobs:
        if k.requires is not None:
            parent, vals = hps[k.requires[0]], [str(v) for v in k.requires[1]]
            cs.add(EqualsCondition(hps[k.name], parent, vals[0]) if len(vals) == 1
                   else InCondition(hps[k.name], parent, vals))
    topo = [k for k in space.knobs if k.name in TOPOLOGY]
    for vs in product(*(k.values for k in topo)):
        if not is_feasible({k.name: v for k, v in zip(topo, vs)}, space.scenario):
            clauses = [ForbiddenEqualsClause(hps[k.name], str(v)) for k, v in zip(topo, vs)]
            cs.add(clauses[0] if len(clauses) == 1 else ForbiddenAndConjunction(*clauses))
    return cs


class SmacSearcher:
    """One trial at a time via ask/tell; SMAC's output files go to a temporary directory.
    trace.extra["stop"] is "budget" or "stall"."""

    def __init__(self, space: KnobSpace, n_init: int = 10, name: str = "smac"):
        self.space, self.n_init, self.name = space, n_init, name

    def search(self, objective, budget: int, seed: int) -> Trace:
        if next(enumerate_feasible(self.space), None) is None:
            raise ValueError(f"no feasible decision in the space for scenario {self.space.scenario}")
        with tempfile.TemporaryDirectory() as tmp:
            sc = Scenario(to_configspace(self.space), deterministic=True, n_trials=1_000_000, seed=seed,
                          output_directory=Path(tmp) / "smac")
            smac = HyperparameterOptimizationFacade(
                sc, lambda config, seed=0: 0.0, overwrite=True, logging_level=logging.ERROR,
                initial_design=HyperparameterOptimizationFacade.get_initial_design(sc, n_configs=self.n_init))
            return self._loop(smac, objective, budget, seed)

    def _loop(self, smac, objective, budget, seed) -> Trace:
        rec, labels = Recorder(self.space, objective), [tuple(map(str, k.values)) for k in self.space.knobs]
        stall, stop = 0, "budget"
        while objective.runs_used < budget:
            if stall >= MAX_STALL:
                stop = "stall"
                break
            info = smac.ask()
            c = dict(info.config)
            d = tuple(opts.index(c[k.name]) if k.name in c else None for k, opts in zip(self.space.knobs, labels))
            before = objective.runs_used
            r = rec.record(d, objective(d))
            smac.tell(info, TrialValue(cost=FAILED, status=StatusType.CRASHED) if r.score is None
                      else TrialValue(cost=r.score))
            stall = stall + 1 if objective.runs_used == before else 0
        return Trace(rec.points, self.name, seed, {"stop": stop})
