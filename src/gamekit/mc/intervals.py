"""Confidence intervals: the normal quantile helper, ``MCResult``, and the
Wilson score interval for a binomial proportion.

Stdlib-only. Extracted verbatim from catan's ``experiments/mcstats.py``, which
was itself written against mmo-utils' naming (see ``mc/__init__.py``'s module
docstring for the full lineage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import NamedTuple

_NORMAL = NormalDist()


def _z_score(alpha: float) -> float:
    """z_{alpha/2}: the normal quantile such that P(|Z| <= z) = 1 - alpha.

    Uses ``statistics.NormalDist.inv_cdf`` (exact, stdlib) rather than a
    rational approximation.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return _NORMAL.inv_cdf(1 - alpha / 2)


class ConfidenceInterval(NamedTuple):
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class MCResult:
    """Mean/variance estimate from ``n`` i.i.d. samples."""

    mean: float
    variance: float
    n: int

    def confidence_interval(self, alpha: float = 0.05) -> ConfidenceInterval:
        if self.n < 2:
            raise ValueError("need at least 2 samples for a confidence interval")
        z = _z_score(alpha)
        half_width = z * math.sqrt(self.variance / self.n)
        return ConfidenceInterval(self.mean - half_width, self.mean + half_width)


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion.

    Prefer this over the normal approximation for every proportion: Wilson
    stays inside [0, 1] and has much better coverage at small ``n`` or
    extreme ``p``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    z = _z_score(alpha)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return ConfidenceInterval((center - spread) / denom, (center + spread) / denom)
