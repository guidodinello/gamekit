"""Streaming accumulators over Monte Carlo samples.

``Accumulator[V, S, R]`` is the generalisation mmo-utils' ``DESIGN.md`` (gap 2)
proposed but never shipped: mmo-utils' own ``Accumulator[S, R]`` still locks
``update`` to a raw ``float``, forcing every evaluator to reduce a sample to a
scalar before the accumulator sees it. catan's ``Accumulator[S]`` moved that
projection to construction time (see ``MeanAccumulator`` below) but is still a
mean/variance-only fold -- it cannot express a joint accumulation over, say, a
``(cost, time)`` pair, an order statistic, or a ratio estimator.

This protocol is the real fix: ``update`` folds an arbitrary evaluation output
``V`` into a state ``S``, and ``finalize`` produces any result type ``R`` --
conventionally an ``MCResult`` (or a `dataclass` wrapping one), which is what
gap 3 (no confidence intervals for custom accumulators) resolves: any
accumulator that finalizes into an ``MCResult`` inherits
``MCResult.confidence_interval`` for free, no extra plumbing.

``update``/``finalize`` are declared as Protocol **methods**, not
``Callable``-typed attributes. A generic Protocol's ``Callable``-typed
attribute checks a member's exact call signature against a plain
class-attribute lookup, which is brittle for a generic fold (it rejected a
perfectly ordinary method-based implementation in testing -- a protocol a
plain class can't satisfy without a ``type: ignore`` is a design bug, not a
strictness feature). A ``def update(self, state, value) -> state`` method
matches this protocol the ordinary way any Protocol method does.

``MeanAccumulator`` below is catan's ``Accumulator[S]`` (its Welford
streaming mean/variance implementation), renamed to free the ``Accumulator``
name for the protocol above. Behaviour is unchanged, but it does **not**
structurally satisfy ``Accumulator[V, S, R]`` -- its ``update`` mutates
``self`` and returns ``None`` rather than folding an explicit state, and it
has no ``init``/``finalize``. It is kept as its own concrete, ergonomic
mean/variance accumulator, not as "the ``V = float`` case" of the protocol
above.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from gamekit.mc.intervals import MCResult


@runtime_checkable
class Accumulator[V, S, R](Protocol):
    """A fold from a stream of evaluation outputs ``V`` to a result ``R``,
    via an intermediate state ``S``.

    ``init`` is the accumulator's zero state, ``update`` folds one sample's
    evaluation output in, and ``finalize`` reads the current state out as a
    result -- conventionally ``MCResult`` so callers get a confidence interval
    for free.
    """

    init: S

    def update(self, state: S, value: V) -> S:
        """Fold one evaluation output into ``state``, returning the next
        state. Pure: does not mutate ``state`` in place."""
        ...

    def finalize(self, state: S) -> R:
        """Read a final result out of a state."""
        ...


@dataclass(slots=True)
class MeanAccumulator[S]:
    """Welford streaming mean/variance accumulator over samples of any type
    ``S``, via a ``value`` projection to ``float``.

    The projection is supplied once, at construction, and every ``update``
    call takes the raw sample (a game record, a per-vertex outcome, whatever).
    This is a concrete, self-mutating convenience accumulator -- it does not
    itself implement the ``Accumulator[V, S, R]`` protocol above (see that
    protocol's docstring): its ``update`` mutates ``self`` and returns
    ``None``, and it has no ``init``/``finalize``.
    """

    value: Callable[[S], float]
    count: int = 0
    _mean: float = 0.0
    _m2: float = 0.0

    def update(self, sample: S) -> None:
        x = self.value(sample)
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def update_all(self, samples: Iterable[S]) -> None:
        for sample in samples:
            self.update(sample)

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ZeroDivisionError("no samples accumulated")
        return self._mean

    @property
    def variance(self) -> float:
        """Sample variance, Bessel-corrected (ddof=1)."""
        if self.count < 2:
            raise ZeroDivisionError("need at least 2 samples for variance")
        return self._m2 / (self.count - 1)

    def result(self) -> MCResult:
        return MCResult(mean=self.mean, variance=self.variance, n=self.count)
