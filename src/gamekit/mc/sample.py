"""One Monte Carlo sampling model: ``Sampler -> Evaluator -> Accumulator``.

Resolves mmo-utils' ``DESIGN.md`` gap 1 -- it exposed two incompatible entry
points, a vectorized ``monte_carlo(Fx_generator, n)`` (numpy, batch variance
formula) and an iterative ``monte_carlo_reduce(n, sampler, evaluator,
accumulator)`` (scalar, generic ``Accumulator``). This module unifies them
behind one signature rather than a protocol with two admissible arities:

    type Sampler[T] = Callable[[int], Iterable[T]]

A vectorized generator (``lambda n: rng.uniform(size=n)``, returning a numpy
``ndarray``) satisfies this structurally -- ``ndarray`` is iterable, and mypy
accepts it wherever ``Iterable[float]`` is expected, verified against numpy's
own stubs. A scalar draw wraps via ``scalar_sampler``. Neither case requires
gamekit to import numpy: the vectorized path only ever needs numpy as a *fast
loop* for generation and evaluation, never as a *type* this module names.
This is why gamekit stays stdlib-only through this feature -- no ``[numpy]``
extra, no import-guarded module. numpy becoming a real dependency (or an
extra) was one option the issue raised; the ``Iterable`` signature makes
neither necessary.

**No sampler here ever owns randomness.** ``Sampler[T]`` takes a count and
returns values -- it never takes or generates a seed. Both catan and truco-py
already rely on common random numbers across arms/games (paired seed
schedules that make an arm-to-arm difference low-variance); an API that
seeded internally would silently break that pairing. The caller keeps its own
``random.Random`` (or numpy ``Generator``) and its own seed schedule.

A sampler is called once per chunk (once per ``n``-sized draw request) and
must return a **fresh** iterable each call -- a generator *function* is fine,
a stored, already-partially-consumed generator is not. ``monte_carlo_reduce``
consumes the returned iterable and requires **exactly** ``n`` values: more or
fewer raises ``ValueError``, never a silent truncation or under-count. This
matters beyond bookkeeping -- ``gamekit.mc.variance``'s antithetic wrapper and
``gamekit.mc.stopping``'s adaptive stopper both depend on knowing precisely
how many underlying draws a call consumed.

``monte_carlo`` is a convenience over ``monte_carlo_reduce``, fixed to
``WelfordAccumulator`` (mean/variance, gamekit's ``MCResult`` convention --
see that class's docstring for how its variance differs from mmo-utils'
``mean_accumulator``). Its argument order, ``(n, sampler, evaluator)``,
mirrors ``monte_carlo_reduce``'s rather than mmo-utils' ``(Fx_generator, n)``,
so the convenience/base relationship is visible at the call site. mmo-utils'
pre-evaluated-generator case is ``monte_carlo(n, generator, lambda x: x)``.

Not ported from mmo-utils: ``monte_carlo_joint`` (a fixed-``k``-column mean is
exactly what a custom ``Accumulator`` already expresses -- gap 2 is fixed,
this would just be reinventing it), ``empirical_coverage``,
``confidence_interval_chebyshev``, ``confidence_interval_agresti_coull``
(``gamekit.mc.wilson_interval`` already covers the bounded-proportion case,
with better small-``n`` coverage).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from gamekit.mc.accumulate import Accumulator
from gamekit.mc.intervals import MCResult

type Sampler[T] = Callable[[int], Iterable[T]]
"""Draws ``n`` values of type ``T``. Called once per chunk; must return a
fresh iterable each call. Owns no randomness -- the caller supplies and
controls it. A vectorized generator returning a numpy ``ndarray`` satisfies
this structurally; see this module's docstring."""

type Evaluator[T, V] = Callable[[T], V]
"""Maps one drawn value ``T`` to one evaluation output ``V``, which the
accumulator folds. ``V`` need not be ``float`` -- see ``Accumulator[V, S, R]``
in ``gamekit.mc.accumulate``."""


def scalar_sampler[T](draw: Callable[[], T]) -> Sampler[T]:
    """Wrap a one-at-a-time draw function (e.g. ``rng.random``) as a
    ``Sampler[T]`` -- the iterative side of the unification."""

    def sample(n: int) -> Iterable[T]:
        return (draw() for _ in range(n))

    return sample


@dataclass(frozen=True, slots=True)
class WelfordState:
    """Immutable running-mean/variance state (Welford's algorithm)."""

    n: int
    mean: float
    m2: float


class WelfordAccumulator:
    """Streaming mean/variance fold satisfying ``Accumulator[float,
    WelfordState, MCResult]`` -- the protocol-satisfying counterpart to
    ``MeanAccumulator`` (``gamekit.mc.accumulate``), which does not itself
    satisfy the protocol (see that class's docstring). ``MeanAccumulator``
    stays the ergonomic, self-mutating accumulator you feed by hand;
    ``WelfordAccumulator`` is the pure fold ``monte_carlo``/
    ``monte_carlo_reduce`` drive.

    A plain class with a class-level ``init`` attribute, not a dataclass --
    matching ``_SumBoth`` in ``tests/test_mc.py``, the existing proof that an
    ordinary class satisfies this generic protocol with no ``type: ignore``.

    ``finalize``'s variance is ``m2 / (n - 1)`` -- the **sample** variance,
    gamekit's ``MCResult`` convention. This deliberately does not match
    mmo-utils' ``mean_accumulator``, whose ``V_hat = M2 / (n * (n - 1))`` is
    the variance *of the estimator*: ``MCResult.confidence_interval`` already
    divides by ``n`` itself, so porting mmo-utils' formula verbatim would
    narrow every confidence interval computed through this fold by a factor
    of sqrt(n).
    """

    init: WelfordState = WelfordState(0, 0.0, 0.0)

    def update(self, state: WelfordState, value: float) -> WelfordState:
        n = state.n + 1
        delta = value - state.mean
        mean = state.mean + delta / n
        m2 = state.m2 + delta * (value - mean)
        return WelfordState(n=n, mean=mean, m2=m2)

    def finalize(self, state: WelfordState) -> MCResult:
        if state.n < 2:
            raise ValueError("need at least 2 samples for a variance estimate")
        return MCResult(mean=state.mean, variance=state.m2 / (state.n - 1), n=state.n)


def _drawn[T](sampler: Sampler[T], n: int) -> list[T]:
    """Consume ``sampler(n)`` and require exactly ``n`` values -- the
    iterable-consumption contract this module's docstring commits to."""
    values = list(sampler(n))
    if len(values) != n:
        raise ValueError(
            f"sampler(n={n}) returned {len(values)} values, expected exactly {n}"
        )
    return values


def monte_carlo_reduce[T, V, S, R](
    n: int,
    sampler: Sampler[T],
    evaluator: Evaluator[T, V],
    accumulator: Accumulator[V, S, R],
) -> R:
    """The base of gamekit's sampling model: draw ``n`` samples, evaluate
    each, and fold the evaluation outputs through ``accumulator``.

        state = accumulator.init
        for value in (evaluator(x) for x in sampler(n)):
            state = accumulator.update(state, value)
        return accumulator.finalize(state)

    Any estimand ``accumulator`` can express is available here -- not just a
    mean (see ``monte_carlo`` for that convenience case). ``n`` must be
    positive.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    state = accumulator.init
    for x in _drawn(sampler, n):
        state = accumulator.update(state, evaluator(x))
    return accumulator.finalize(state)


def monte_carlo[T](
    n: int, sampler: Sampler[T], evaluator: Evaluator[T, float]
) -> MCResult:
    """Estimate a mean via ``monte_carlo_reduce`` with ``WelfordAccumulator``
    -- the convenience shortcut mmo-utils' ``DESIGN.md`` proposed. Equivalent
    to ``monte_carlo_reduce(n, sampler, evaluator, WelfordAccumulator())``.
    """
    return monte_carlo_reduce(n, sampler, evaluator, WelfordAccumulator())
