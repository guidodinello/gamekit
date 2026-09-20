"""Two-proportion hypothesis testing and Benjamini-Hochberg FDR control.
Extracted verbatim from catan's ``experiments/mcstats.py``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from gamekit.mc.intervals import _NORMAL


class TwoProportionTest(NamedTuple):
    z: float
    p_value: float


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> TwoProportionTest:
    """Two-sided pooled z-test for a difference between two proportions
    (``k1`` successes of ``n1`` vs. ``k2`` of ``n2``)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive")
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return TwoProportionTest(z=0.0, p_value=1.0)
    z = (p1 - p2) / se
    p_value = 2 * (1 - _NORMAL.cdf(abs(z)))
    return TwoProportionTest(z=z, p_value=p_value)


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg false-discovery-rate control across many pairwise
    comparisons. Returns a same-length, same-order list of booleans: True
    where the null is rejected at FDR level ``q``.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    threshold_rank = 0
    for rank, i in enumerate(order, start=1):
        if p_values[i] <= (rank / n) * q:
            threshold_rank = rank
    cutoff = p_values[order[threshold_rank - 1]] if threshold_rank else -1.0
    return [p <= cutoff for p in p_values]
