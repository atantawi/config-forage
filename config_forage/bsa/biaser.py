"""Biasers: experts whose factors multiply the generator's marginals (BSA product of experts).

Extracted from ai-forage c4339d2; this copy is canonical."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from config_forage.bsa.space import Decision, DecisionSpace


class Biaser(Protocol):
    def factor(self, axis: int, partial: Decision) -> np.ndarray | None:
        """Non-negative weights over the options of `axis`, given the options chosen so far.

        A zero weight excludes an option (a hard constraint). None means the axis does not apply
        given `partial`; it is then left unlabelled (None in the decision). `partial` may hold None.
        """
        ...


class IdentityBiaser:
    def __init__(self, space: DecisionSpace):
        self.space = space

    def factor(self, axis: int, partial: Decision) -> np.ndarray:
        return np.ones(self.space.sizes[axis])


class OracleBiaser:
    """Cheating biaser for capability probes: favours a known-good decision vector.

    factor = exp(beta) for the target option on each axis, 1 elsewhere, i.e. B^beta with
    B = e at the target, as in BSA's B(r)^{w_r}.
    """

    def __init__(self, space: DecisionSpace, target: Decision, beta: float):
        self.space, self.target, self.beta = space, tuple(target), beta

    def factor(self, axis: int, partial: Decision) -> np.ndarray:
        f = np.ones(self.space.sizes[axis])
        f[self.target[axis]] = np.exp(self.beta)
        return f


class ApplicabilityBiaser:
    """Structural biaser: an axis applies only when an earlier axis took one of given options.

    `requires` maps axis -> (earlier axis, options of it under which the axis applies). Where an
    axis does not apply, factor is None, so the axis is left unlabelled and the update ignores it.
    Unlabelled earlier axes count as not matching.
    """

    def __init__(self, space: DecisionSpace, requires: dict[int, tuple[int, tuple[int, ...]]]):
        for m, (k, opts) in requires.items():
            if not 0 <= m < len(space.axes):
                raise ValueError(f"axis {m} out of range for a space of {len(space.axes)} axes")
            if not 0 <= k < m:
                raise ValueError(f"axis {m} can only depend on an earlier axis, not {k}")
            if not all(0 <= o < space.sizes[k] for o in opts):
                raise ValueError(f"axis {m}: options {opts} out of range for axis {k}")
        self.space, self.requires = space, {m: (k, tuple(opts)) for m, (k, opts) in requires.items()}

    def factor(self, axis: int, partial: Decision) -> np.ndarray | None:
        if axis in self.requires:
            k, opts = self.requires[axis]
            if partial[k] not in opts:
                return None
        return np.ones(self.space.sizes[axis])
