"""Sample-size calculations for a [0, 1]-bounded mean or a two-proportion
comparison. Extracted verbatim from catan's ``experiments/mcstats.py``.
"""

from __future__ import annotations

import math

from gamekit.mc.intervals import _NORMAL, _z_score


def sample_size_clt(eps: float, delta: float) -> int:
    """Worst-case (p=0.5) sample size so a two-sided (1-delta) CI on a
    [0, 1]-bounded mean has half-width <= eps, via the normal approximation.

    n >= (z_{delta/2} / (2 eps))^2
    """
    z = _z_score(delta)
    return math.ceil((z / (2 * eps)) ** 2)


def sample_size_hoeffding(eps: float, delta: float) -> int:
    """Distribution-free sample size (Hoeffding's inequality) for a
    [0, 1]-bounded variable, no normality assumption -- more conservative
    than ``sample_size_clt``.

    n >= log(2/delta) / (2 eps^2)
    """
    return math.ceil(math.log(2 / delta) / (2 * eps**2))


def two_proportion_sample_size(
    p: float, delta: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Per-arm sample size to detect a difference of ``delta`` between two
    proportions both near ``p``, at significance ``alpha`` and the given
    ``power`` (normal approximation, equal-n two-sample test).
    """
    z_alpha = _z_score(alpha)
    z_power = _NORMAL.inv_cdf(power)
    variance_term = 2 * p * (1 - p)
    return math.ceil(((z_alpha + z_power) ** 2 * variance_term) / delta**2)
