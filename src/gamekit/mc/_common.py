"""Helpers shared by several ``gamekit.mc`` submodules.

Package-internal: nothing here is public or re-exported from ``gamekit.mc``.
Stdlib-only, and imports nothing else from ``gamekit``, so any ``mc`` submodule
can depend on it without creating a cycle (#51).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from statistics import NormalDist

NORMAL = NormalDist()


def z_score(alpha: float) -> float:
    """z_{alpha/2}: the normal quantile such that P(|Z| <= z) = 1 - alpha.

    Uses ``statistics.NormalDist.inv_cdf`` (exact, stdlib) rather than a
    rational approximation.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return NORMAL.inv_cdf(1 - alpha / 2)


def drawn[T](sampler: Callable[[int], Iterable[T]], n: int) -> list[T]:
    """Consume ``sampler(n)`` and require exactly ``n`` values -- the
    iterable-consumption contract ``gamekit.mc.sample``'s docstring commits to.
    ``sampler`` is a ``gamekit.mc.sample.Sampler[T]``, spelled out structurally
    so this module stays a leaf."""
    values = list(sampler(n))
    if len(values) != n:
        raise ValueError(
            f"sampler(n={n}) returned {len(values)} values, expected exactly {n}"
        )
    return values
