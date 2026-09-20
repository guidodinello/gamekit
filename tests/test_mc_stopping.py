"""Tests for gamekit.mc.stopping (mmo-utils DESIGN.md gap 5)."""

from __future__ import annotations

import random
from statistics import NormalDist

import pytest

from gamekit.mc import monte_carlo_until
from gamekit.mc.sample import Sampler

_Z_95 = NormalDist().inv_cdf(1 - 0.05 / 2)


def uniform_sampler(rng: random.Random) -> Sampler[float]:
    def sample(n: int) -> list[float]:
        return [rng.random() for _ in range(n)]

    return sample


def bernoulli_sampler(rng: random.Random, p: float) -> Sampler[float]:
    def sample(n: int) -> list[float]:
        return [1.0 if rng.random() < p else 0.0 for _ in range(n)]

    return sample


def test_stops_near_the_analytic_sample_size_for_a_known_uniform_variance() -> None:
    """X ~ U(0, 1): Var(X) = 1/12 exactly. The analytic n for a target
    half-width h under the normal approximation is (z*sigma/h)^2."""
    sigma = (1 / 12) ** 0.5
    target = 0.01
    analytic_n = (_Z_95 * sigma / target) ** 2

    outcome = monte_carlo_until(
        uniform_sampler(random.Random(0)),
        lambda x: x,
        target_half_width=target,
        max_n=50_000,
        min_n=30,
        batch=200,
    )

    assert outcome.target_met is True
    assert 0.75 * analytic_n <= outcome.result.n <= 1.25 * analytic_n
    assert outcome.result.confidence_interval().width / 2 <= target


def test_stops_near_the_analytic_sample_size_for_a_known_bernoulli_variance() -> None:
    """X ~ Bernoulli(0.5): Var(X) = 0.25 exactly. A second, coarser-target
    known-variance distribution, checked the same way."""
    sigma = 0.5
    target = 0.02
    analytic_n = (_Z_95 * sigma / target) ** 2

    outcome = monte_carlo_until(
        bernoulli_sampler(random.Random(1), 0.5),
        lambda x: x,
        target_half_width=target,
        max_n=50_000,
        min_n=30,
        batch=200,
    )

    assert outcome.target_met is True
    assert 0.75 * analytic_n <= outcome.result.n <= 1.25 * analytic_n
    assert outcome.result.confidence_interval().width / 2 <= target


def test_too_tight_a_target_against_a_small_max_n_does_not_raise() -> None:
    outcome = monte_carlo_until(
        uniform_sampler(random.Random(2)),
        lambda x: x,
        target_half_width=1e-12,
        max_n=500,
        min_n=30,
        batch=100,
    )

    assert outcome.target_met is False
    assert outcome.result.n == 500


def test_min_n_defers_the_first_stopping_check() -> None:
    """Every sample is identical (variance exactly 0) -- the half-width is
    0 from the very first sample, but min_n must still force waiting: the
    first opportunity to stop is the first chunk boundary at or past
    min_n, not before."""

    def constant_sampler(n: int) -> list[float]:
        return [0.5] * n

    outcome = monte_carlo_until(
        constant_sampler,
        lambda x: x,
        target_half_width=1e-9,
        max_n=10_000,
        min_n=500,
        batch=100,
    )

    assert outcome.target_met is True
    assert outcome.result.n == 500


def test_max_n_below_min_n_raises() -> None:
    with pytest.raises(ValueError, match="must be >="):
        monte_carlo_until(
            uniform_sampler(random.Random(0)),
            lambda x: x,
            target_half_width=0.01,
            max_n=10,
            min_n=30,
        )


def test_non_positive_target_half_width_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        monte_carlo_until(
            uniform_sampler(random.Random(0)),
            lambda x: x,
            target_half_width=0.0,
            max_n=1_000,
        )


def test_min_n_below_two_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        monte_carlo_until(
            uniform_sampler(random.Random(0)),
            lambda x: x,
            target_half_width=0.01,
            max_n=1_000,
            min_n=1,
        )


def test_non_positive_batch_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        monte_carlo_until(
            uniform_sampler(random.Random(0)),
            lambda x: x,
            target_half_width=0.01,
            max_n=1_000,
            batch=0,
        )


def test_never_draws_past_max_n() -> None:
    calls: list[int] = []
    rng = random.Random(0)

    def counting_sampler(n: int) -> list[float]:
        calls.append(n)
        return [rng.random() for _ in range(n)]

    outcome = monte_carlo_until(
        counting_sampler,
        lambda x: x,
        target_half_width=1e-12,
        max_n=350,
        min_n=30,
        batch=100,
    )

    assert sum(calls) == 350
    assert calls == [100, 100, 100, 50]
    assert outcome.result.n == 350
