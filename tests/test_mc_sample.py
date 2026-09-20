"""Tests for the unified Sampler -> Evaluator -> Accumulator fold
(``gamekit.mc.sample``), resolving mmo-utils' DESIGN.md gap 1."""

from __future__ import annotations

import math
import random
import statistics

import pytest

from gamekit.mc import (
    Accumulator,
    monte_carlo,
    monte_carlo_reduce,
    scalar_sampler,
)
from gamekit.mc.sample import Sampler, WelfordAccumulator, WelfordState

# True value of the integral 0..1 of exp(-x^2) dx -- the same integrand
# mmo-utils' examples/mc_integration.ipynb uses to demonstrate antithetic
# variates. Also exercised by test_mc_variance.py and test_mc_stopping.py.
TRUE_GAUSSIAN_INTEGRAL = math.sqrt(math.pi) / 2 * math.erf(1.0)


def uniform_sampler(rng: random.Random) -> Sampler[float]:
    """A batch ``Sampler[float]`` drawing from U(0, 1)."""

    def sample(n: int) -> list[float]:
        return [rng.random() for _ in range(n)]

    return sample


def test_monte_carlo_recovers_uniform_mean_and_matches_statistics_variance() -> None:
    rng = random.Random(42)
    n = 5_000
    draws = [rng.random() for _ in range(n)]

    rng_for_mc = random.Random(42)
    result = monte_carlo(n, uniform_sampler(rng_for_mc), lambda x: x)

    assert round(result.mean, 9) == round(statistics.mean(draws), 9)
    assert round(result.variance, 9) == round(statistics.variance(draws), 9)
    ci = result.confidence_interval()
    assert ci.lower < 0.5 < ci.upper


def test_monte_carlo_is_reduce_with_welford_accumulator_on_identical_seeds() -> None:
    n = 1_000

    via_convenience = monte_carlo(n, uniform_sampler(random.Random(7)), lambda x: x)
    via_reduce = monte_carlo_reduce(
        n, uniform_sampler(random.Random(7)), lambda x: x, WelfordAccumulator()
    )

    assert via_convenience == via_reduce


def test_batch_sampler_and_scalar_sampler_agree_on_identical_draws() -> None:
    """The unification claim: a batch sampler and a scalar-draw sampler,
    wrapped via scalar_sampler, produce the same result on the same values."""
    n = 500

    batch_result = monte_carlo(n, uniform_sampler(random.Random(11)), lambda x: x)

    rng = random.Random(11)
    scalar_result = monte_carlo(n, scalar_sampler(rng.random), lambda x: x)

    assert batch_result == scalar_result


def test_monte_carlo_estimates_the_gaussian_integral() -> None:
    rng = random.Random(2026)

    def sample(n: int) -> list[float]:
        return [rng.random() for _ in range(n)]

    result = monte_carlo(100_000, sample, lambda u: math.exp(-(u**2)))

    ci = result.confidence_interval()
    assert ci.lower <= TRUE_GAUSSIAN_INTEGRAL <= ci.upper


class _RatioAccumulator:
    """Models truco-py's ``simular_generico``: a filtered ratio estimator
    (successes / occurrences passing a filter), not a mean -- proof that
    ``monte_carlo_reduce``'s result type ``R`` is genuinely free."""

    init: tuple[int, int] = (0, 0)  # (successes, filtered_n)

    def update(
        self, state: tuple[int, int], value: tuple[bool, bool]
    ) -> tuple[int, int]:
        success, passes_filter = value
        if not passes_filter:
            return state
        successes, filtered_n = state
        return (successes + (1 if success else 0), filtered_n + 1)

    def finalize(self, state: tuple[int, int]) -> float:
        successes, filtered_n = state
        if filtered_n == 0:
            return 0.0
        return successes / filtered_n


def test_monte_carlo_reduce_drives_a_non_mean_ratio_accumulator() -> None:
    rng = random.Random(3)

    def sample(n: int) -> list[float]:
        return [rng.random() for _ in range(n)]

    # Estimate P(X > 0.5 | X > 0.25) for X ~ U(0, 1), true value = 1/1.5.
    def evaluate(x: float) -> tuple[bool, bool]:
        return (x > 0.5, x > 0.25)

    acc: Accumulator[tuple[bool, bool], tuple[int, int], float] = _RatioAccumulator()
    result = monte_carlo_reduce(50_000, sample, evaluate, acc)

    assert isinstance(result, float)
    assert abs(result - (1 / 1.5)) < 0.02


def test_sampler_returning_too_few_values_raises() -> None:
    def short_sampler(n: int) -> list[float]:
        return [0.5] * (n - 1)

    with pytest.raises(ValueError, match="expected exactly"):
        monte_carlo(10, short_sampler, lambda x: x)


def test_sampler_returning_too_many_values_raises() -> None:
    def long_sampler(n: int) -> list[float]:
        return [0.5] * (n + 1)

    with pytest.raises(ValueError, match="expected exactly"):
        monte_carlo(10, long_sampler, lambda x: x)


def test_sampler_is_consumed_exactly_n_times_not_truncated() -> None:
    calls: list[int] = []

    def counting_sampler(n: int) -> list[float]:
        calls.append(n)
        return [0.5] * n

    monte_carlo(37, counting_sampler, lambda x: x)
    assert calls == [37]


def test_monte_carlo_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="positive"):
        monte_carlo(0, uniform_sampler(random.Random(0)), lambda x: x)


def test_welford_accumulator_satisfies_the_accumulator_protocol() -> None:
    acc: Accumulator[float, WelfordState, object] = WelfordAccumulator()
    assert isinstance(acc, Accumulator)


def test_welford_accumulator_raises_below_two_samples() -> None:
    acc = WelfordAccumulator()
    state = acc.update(acc.init, 1.0)
    with pytest.raises(ValueError, match="at least 2 samples"):
        acc.finalize(state)
