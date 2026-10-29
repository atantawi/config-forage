"""Categorical generator P over a DecisionSpace with a smoothed elite (CEM/BSA) update.

Extracted from the author's earlier BSA research code (commit c4339d2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from config_forage.bsa.biaser import Biaser
from config_forage.bsa.space import Decision, DecisionSpace


class InfeasibleSample(RuntimeError):
    """The biasers left no feasible option on some axis, in every try."""


Scheme = Literal["hard", "boltzmann", "rank"]


def default_eps(space: DecisionSpace, eps: float = 0.02) -> float:
    """eps, capped at half the uniform mass of the widest axis: min(eps, 0.5 / max axis size). The
    update P = eps + (1 - n*eps) q needs eps * n < 1 on every axis, so a wide axis (100 options, say)
    still works with the default. Any space whose widest axis has at most 25 options keeps eps."""
    return min(eps, 0.5 / max(space.sizes, default=1))


class CategoricalGenerator:
    """`scheme` picks the elite update:

    - "hard": the best rho fraction, equal weights (the original update);
    - "boltzmann": the samples at or better than the q-quantile, weighted exp(-(s - s_best)/tau)
      with tau = c * MAD of their scores (equal weights when the MAD is 0). If one sample far
      ahead of the rest leaves fewer than `min_ess` effective elites, tau is doubled until it
      does not, so a lucky realization cannot take the whole update;
    - "rank": the same group, CMA-ES rank weights log((mu + 1/2)/i), averaged over tied scores.
    """

    def __init__(self, space: DecisionSpace, alpha: float = 0.7, rho: float = 0.3, eps: float = 0.02,
                 scheme: Scheme = "hard", q: float = 0.5, c: float = 1.0, min_ess: float = 2.0):
        if scheme not in ("hard", "boltzmann", "rank"):
            raise ValueError(f"unknown scheme {scheme!r}")
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], not {alpha}")
        if not 0 < rho <= 1:
            raise ValueError(f"rho must be in (0, 1], not {rho}")
        if not eps >= 0:
            raise ValueError(f"eps must be non-negative, not {eps}")
        for a, n in zip(space.axes, space.sizes):  # P = eps + (1 - n*eps) q is a distribution iff eps * n < 1
            if not eps * n < 1:
                raise ValueError(f"eps {eps} is too large for axis {a.name!r} with {n} options: eps * {n} must "
                                 f"be < 1, so eps < {1 / n:.6g} (default_eps(space) gives {default_eps(space)})")
        if not 0 < q <= 1:
            raise ValueError(f"q must be in (0, 1], not {q}")
        if not c > 0:
            raise ValueError(f"c must be positive, not {c}")
        if not min_ess >= 1:
            raise ValueError(f"min_ess must be at least 1, not {min_ess}")
        self.space, self.alpha, self.rho, self.eps = space, alpha, rho, eps
        self.scheme, self.q, self.c, self.min_ess = scheme, q, c, min_ess
        self.P = [np.full(n, 1.0 / n) for n in space.sizes]
        self.last_weights: list[float] = []  # of the last update's elites, normalized to sum 1

    def sample(self, rng: np.random.Generator, biasers: Sequence[Biaser] = (), max_tries: int = 10) -> Decision:
        """Build a decision axis by axis, each conditioned on the options already chosen.

        A biaser factor of None marks the axis as not applicable: it is left unlabelled (None) and
        the remaining biasers are not asked, so put cheap structural biasers before an LLM prior.
        If the factors leave an axis no option, the sample is a dead end and is started again, up
        to `max_tries` times (as in bsa_c); then InfeasibleSample is raised.

        Each sample takes exactly one uniform per axis from `rng`, whatever the biasers answer, so
        a resumed run that samples again stays in step even if a biaser answers differently. The
        draws match `rng.choice(n, p=w)` per axis. Weights that are negative, not finite or overflow
        when summed raise ValueError (an OracleBiaser with a huge beta, say), as rng.choice did. Retries draw from a child generator seeded from
        those uniforms.
        """
        if max_tries < 1:
            raise ValueError(f"max_tries must be at least 1, not {max_tries}")
        u = rng.random(len(self.P))
        d = self._try_sample(u, biasers)
        if d is not None:
            return d
        retry = np.random.default_rng(np.floor(u * 2**32).astype(np.uint64))
        for _ in range(max_tries - 1):
            d = self._try_sample(retry.random(len(self.P)), biasers)
            if d is not None:
                return d
        raise InfeasibleSample(f"no feasible decision in {max_tries} tries")

    def _try_sample(self, u: np.ndarray, biasers: Sequence[Biaser]) -> Decision | None:
        d: list[int | None] = []
        for m, p in enumerate(self.P):
            w = p.copy()
            for b in biasers:
                f = b.factor(m, tuple(d))
                if f is None:
                    w = None
                    break
                w = w * f
            if w is None:
                d.append(None)
                continue
            if not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError(f"axis {m}: biaser weights must be finite and non-negative, got {w}")
            total = w.sum()
            if not np.isfinite(total):
                raise ValueError(f"axis {m}: biaser weights overflow (sum is {total})")
            if not total > 0:
                return None
            cdf = (w / total).cumsum()  # as in Generator.choice
            cdf /= cdf[-1]
            d.append(int(cdf.searchsorted(u[m], side="right")))
        return tuple(d)

    def update(self, decisions: Sequence[tuple[int | None, ...]], scores: Sequence[float | None]) -> list[int]:
        """Move P toward the elites of the successful samples (lower score is better).

        Returns the indices of the elites, best first; their weights are in `last_weights`. Failed
        samples (score None) are never elites; if all failed, P is unchanged. A decision may leave
        an axis unlabelled (None): on that axis the elite frequencies use only the labelled elites,
        with their weights renormalized, and with none the axis is unchanged.
        """
        ok = [i for i, s in enumerate(scores) if s is not None]
        if not ok:
            self.last_weights = []
            return []
        ok.sort(key=lambda i: scores[i])  # stable: ties go to the earlier sample
        elites, w = self._elites(ok, np.array([scores[i] for i in ok], dtype=float))
        self.last_weights = (w / w.sum()).tolist()
        for m, p in enumerate(self.P):
            labelled = [(decisions[i][m], wi) for i, wi in zip(elites, w) if decisions[i][m] is not None]
            if not labelled:
                continue
            opts, ws = zip(*labelled)
            freq = np.bincount(opts, weights=ws, minlength=len(p)) / sum(ws)
            q = (1 - self.alpha) * p + self.alpha * freq
            self.P[m] = self.eps + (1 - len(p) * self.eps) * q
        return elites

    def _elites(self, ok: list[int], s: np.ndarray) -> tuple[list[int], np.ndarray]:
        """The elites among `ok` (sorted best first, scores `s`) and their unnormalized weights."""
        if self.scheme == "hard":
            mu = max(1, round(self.rho * len(ok)))
            return ok[:mu], np.ones(mu)
        if not np.all(np.isfinite(s)):
            raise ValueError(f"scores must be finite for the {self.scheme} scheme, got {s}")
        mu = int((s <= np.quantile(s, self.q)).sum())  # ties at the cut are all in
        s = s[:mu]
        if self.scheme == "boltzmann":
            s = s / max(np.abs(s).max(), np.finfo(float).tiny)  # the weights are scale-free; no overflow
            mad = np.median(np.abs(s - np.median(s)))
            if mad == 0:
                return ok[:mu], np.ones(mu)
            tau = max(self.c * mad, np.finfo(float).tiny)  # c * mad can underflow to 0, and 0/0 is NaN
            w = np.exp(-(s - s[0]) / tau)
            while _ess(w) < min(self.min_ess, mu) - 1e-9:  # equal weights (ESS = mu) as tau grows
                tau *= 2
                w = np.exp(-(s - s[0]) / tau)
        else:
            r = np.log((mu + 0.5) / np.arange(1, mu + 1))
            w = np.array([r[s == x].mean() for x in s])  # tied scores share their ranks' weight
        return ok[:mu], w

    @property
    def ess(self) -> float:
        """Effective number of elites in the last update, (sum w)^2 / sum w^2; 0 before any."""
        return _ess(np.array(self.last_weights)) if self.last_weights else 0.0

    def entropy(self) -> float:
        return float(sum(-(p * np.log(p)).sum() for p in self.P))

    def snapshot(self) -> list[list[float]]:
        return [p.tolist() for p in self.P]


def _ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())
