"""Adaptive stopping: run until a target confidence-interval half-width.

Resolves mmo-utils' ``DESIGN.md`` gap 5. Every sampling entry point in
mmo-utils (and in ``gamekit`` up to this point) required ``n`` upfront, with
no way to stop early once an estimate is precise enough, or to keep going
past a hand-picked ``n`` that turned out to be too small. Built directly on
gap 1's fold: ``monte_carlo_until`` draws in chunks through the same
``WelfordAccumulator`` ``monte_carlo``/``monte_carlo_reduce`` use, checking
the running confidence-interval half-width after each chunk once at least
``min_n`` samples have been drawn.

``max_n`` is a required keyword, not a default -- there is no unbounded
loop. Hitting it returns ``target_met=False`` with whatever estimate was
accumulated rather than raising, so a caller always gets *something* usable
(the caller decides whether an unmet target is acceptable). ``min_n``
(default 30, matching the informal normal-approximation threshold used
throughout ``gamekit.mc.sample_size``) exists because a chunk of unluckily
similar early samples can otherwise report a spuriously tiny variance and
stop after only 2 samples, before the estimate has any real precision.

**Kept deliberately concrete, not generic over ``Accumulator``.** A stopping
rule needs the running standard error, which only ``finalize`` exposes on
the ``Accumulator[V, S, R]`` protocol -- there is no way to probe ``S``
directly without committing to a specific ``R``. Rather than add a generic
``probe: Callable[[S], MCResult]`` parameter that nobody has a non-mean use
for yet, ``monte_carlo_until`` is fixed to ``WelfordAccumulator`` /
``MCResult``, exactly like ``monte_carlo``. If a non-mean stopping rule is
ever needed, a generic version is a strictly additive follow-up.

**Not wired into ``gamekit.benchmark.run_arm``.** ``run_arm`` requires
``n_games`` to be an exact multiple of the lineup length for its seat
rotation to be exact (see that module's docstring), so an adaptive stopper
can never be dropped in without also stopping only on lineup-length
multiples -- a real constraint, left as known future work rather than
designed here on a guess. See the DESIGN discussion for gap 5's issue.
"""

from __future__ import annotations

from dataclasses import dataclass

from gamekit.mc.intervals import MCResult
from gamekit.mc.sample import Evaluator, Sampler, WelfordAccumulator, _drawn


@dataclass(frozen=True, slots=True)
class AdaptiveResult:
    """The outcome of ``monte_carlo_until``: the ``MCResult`` accumulated so
    far, and whether the target half-width was actually reached
    (``False`` iff ``max_n`` was hit first)."""

    result: MCResult
    target_met: bool


def monte_carlo_until[T](
    sampler: Sampler[T],
    evaluator: Evaluator[T, float],
    *,
    target_half_width: float,
    max_n: int,
    alpha: float = 0.05,
    min_n: int = 30,
    batch: int = 1_000,
) -> AdaptiveResult:
    """Draw from ``sampler`` in chunks of ``batch``, evaluating and folding
    each through ``WelfordAccumulator``, until the ``(1 - alpha)``
    confidence-interval half-width drops to ``target_half_width`` or ``n``
    reaches ``max_n`` -- whichever comes first. The half-width is only
    checked once ``n >= min_n``.

    Never draws past ``max_n``: the last chunk is shrunk so ``n`` lands
    exactly on ``max_n`` rather than overshooting it.
    """
    if target_half_width <= 0:
        raise ValueError(f"target_half_width must be positive, got {target_half_width}")
    if min_n < 2:
        raise ValueError(f"min_n must be at least 2, got {min_n}")
    if max_n < min_n:
        raise ValueError(f"max_n ({max_n}) must be >= min_n ({min_n})")
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")

    accumulator = WelfordAccumulator()
    state = accumulator.init
    n = 0
    while n < max_n:
        draw = min(batch, max_n - n)
        for x in _drawn(sampler, draw):
            state = accumulator.update(state, evaluator(x))
        n += draw
        if n >= min_n:
            result = accumulator.finalize(state)
            half_width = result.confidence_interval(alpha).width / 2
            if half_width <= target_half_width:
                return AdaptiveResult(result=result, target_met=True)

    return AdaptiveResult(result=accumulator.finalize(state), target_met=False)
