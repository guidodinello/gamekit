"""Tests for gamekit.mc.variance (mmo-utils DESIGN.md gap 4).

Every variance claim here is checked at an *equal draw budget* --
result.variance / result.n, i.e. the variance of the estimator -- never a
bare comparison of MCResult.variance. A paired-mean sample has smaller
spread than a raw sample regardless of correlation, which would look like
variance reduction even with no real effect; the negative-control tests
(a self-pairing reflection, and an independent-second-draw reflection)
exist specifically to separate genuine negative correlation from the
arithmetic of averaging. See gamekit/mc/variance.py's module docstring for
the exact math each control is checking."""

from __future__ import annotations

import math
import random

import pytest

from gamekit.mc import (
    antithetic,
    control_beta,
    control_variate,
    monte_carlo,
    paired_mean,
)
from gamekit.mc.sample import Sampler

TRUE_GAUSSIAN_INTEGRAL = math.sqrt(math.pi) / 2 * math.erf(1.0)


def uniform_sampler(rng: random.Random) -> Sampler[float]:
    def sample(n: int) -> list[float]:
        return [rng.random() for _ in range(n)]

    return sample


def _estimator_variance(variance: float, n: int) -> float:
    return variance / n


def test_antithetic_reduces_estimator_variance_on_gaussian_integral() -> None:
    """True antithetic reflection (u -> 1-u) on integral_0^1 exp(-x^2) dx:
    at an equal draw budget (n plain uniforms vs n/2 antithetic pairs, both
    evaluating 2*(n/2)=n points), the estimator variance drops by >= 10x."""
    n = 20_000

    plain = monte_carlo(
        n, uniform_sampler(random.Random(1)), lambda u: math.exp(-(u**2))
    )
    anti = monte_carlo(
        n // 2,
        antithetic(uniform_sampler(random.Random(1)), lambda u: 1.0 - u),
        paired_mean(lambda u: math.exp(-(u**2))),
    )

    plain_var = _estimator_variance(plain.variance, plain.n)
    anti_var = _estimator_variance(anti.variance, anti.n)

    assert anti_var > 0
    assert plain_var / anti_var >= 10.0

    for result in (plain, anti):
        ci = result.confidence_interval()
        assert ci.lower <= TRUE_GAUSSIAN_INTEGRAL <= ci.upper


def test_self_pairing_reflection_gives_no_variance_reduction() -> None:
    """Negative control: reflect=identity means every pair is (x, x) --
    perfectly *positively* correlated, so paired_mean(f)(x, x) == f(x)
    exactly. This must NOT look like variance reduction; the estimator
    variance ratio against plain MC at the same draw budget must be ~1."""
    n = 20_000

    plain = monte_carlo(
        n, uniform_sampler(random.Random(5)), lambda u: math.exp(-(u**2))
    )
    self_paired = monte_carlo(
        n // 2,
        antithetic(uniform_sampler(random.Random(5)), lambda u: u),
        paired_mean(lambda u: math.exp(-(u**2))),
    )

    plain_var = _estimator_variance(plain.variance, plain.n)
    self_var = _estimator_variance(self_paired.variance, self_paired.n)

    ratio = plain_var / self_var
    assert 0.5 <= ratio <= 2.0, f"expected ~1x (no reduction), got {ratio:.2f}x"


def test_independent_second_draw_gives_no_estimator_variance_reduction() -> None:
    """Negative control: pairing with an independent second draw (Cov = 0,
    not a true reflection) must give NO reduction in estimator variance --
    at equal total draws, averaging two independent values and using half
    as many paired samples is mathematically identical to using the raw
    values directly (Var(pair average) = sigma^2/2 exactly cancels against
    using n/2 pairs instead of n samples). The ratio must land at ~1x, well
    short of the >= 10x a real negative-correlation reflection achieves
    above, and clearly above the self-pairing case's ~0.5x (below)."""
    n = 20_000
    rng_second = random.Random(999)

    plain = monte_carlo(
        n, uniform_sampler(random.Random(6)), lambda u: math.exp(-(u**2))
    )
    independent_paired = monte_carlo(
        n // 2,
        antithetic(uniform_sampler(random.Random(6)), lambda _u: rng_second.random()),
        paired_mean(lambda u: math.exp(-(u**2))),
    )

    plain_var = _estimator_variance(plain.variance, plain.n)
    indep_var = _estimator_variance(independent_paired.variance, independent_paired.n)

    ratio = plain_var / indep_var
    assert 0.8 <= ratio <= 1.25, f"expected ~1x (no reduction), got {ratio:.2f}x"


def test_antithetic_on_linear_integrand_has_exact_reflection() -> None:
    """integral_0^1 x dx: x + (1-x) == 1 identically, so every paired mean
    is exactly 0.5 -- the estimator variance is (numerically) zero."""
    n = 2_000

    result = monte_carlo(
        n,
        antithetic(uniform_sampler(random.Random(3)), lambda u: 1.0 - u),
        paired_mean(lambda u: u),
    )

    assert result.mean == pytest.approx(0.5, abs=1e-9)
    assert result.variance < 1e-20


def test_control_variate_reduces_estimator_variance_on_exponential_integrand() -> None:
    """integral_0^1 e^x dx, control c(x) = x with the exactly known
    E[c] = 0.5. A pilot sample estimates beta; applying it must reduce the
    estimator variance by >= 10x versus plain MC at the same seeded draws."""
    n = 20_000

    def f(x: float) -> float:
        return math.exp(x)

    def c(x: float) -> float:
        return x

    pilot_rng = random.Random(42)
    pilot = [(f(x), c(x)) for x in (pilot_rng.random() for _ in range(2_000))]
    beta = control_beta(pilot)

    plain = monte_carlo(n, uniform_sampler(random.Random(10)), f)
    adjusted = monte_carlo(
        n,
        uniform_sampler(random.Random(10)),
        control_variate(f, c, control_mean=0.5, beta=beta),
    )

    plain_var = _estimator_variance(plain.variance, plain.n)
    adj_var = _estimator_variance(adjusted.variance, adjusted.n)

    assert plain_var / adj_var >= 10.0
    assert adjusted.mean == pytest.approx(plain.mean, abs=0.01)


def test_control_variate_beta_zero_is_the_identity() -> None:
    def f(x: float) -> float:
        return math.exp(x)

    def c(x: float) -> float:
        return x

    evaluator = control_variate(f, c, control_mean=0.5, beta=0.0)
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert evaluator(x) == f(x)


def test_control_variate_wrong_sign_beta_increases_variance() -> None:
    """Sign check: negating the correct beta must make the adjusted
    estimator's variance *worse* than plain MC, not better -- otherwise the
    suite can't tell whether beta is applied with the right sign."""
    n = 20_000

    def f(x: float) -> float:
        return math.exp(x)

    def c(x: float) -> float:
        return x

    pilot_rng = random.Random(42)
    pilot = [(f(x), c(x)) for x in (pilot_rng.random() for _ in range(2_000))]
    beta = control_beta(pilot)
    assert beta > 0  # Cov(e^x, x) > 0 on [0, 1]

    plain = monte_carlo(n, uniform_sampler(random.Random(11)), f)
    negated = monte_carlo(
        n,
        uniform_sampler(random.Random(11)),
        control_variate(f, c, control_mean=0.5, beta=-beta),
    )

    plain_var = _estimator_variance(plain.variance, plain.n)
    negated_var = _estimator_variance(negated.variance, negated.n)

    assert negated_var > plain_var


def test_control_beta_recovers_analytic_beta_on_a_linear_relationship() -> None:
    """f = 3*c exactly => Cov(f, c) = 3*Var(c) => beta = 3."""
    rng = random.Random(0)
    pilot = [(3.0 * c, c) for c in (rng.random() for _ in range(1_000))]
    assert control_beta(pilot) == pytest.approx(3.0, abs=1e-9)


def test_control_beta_requires_at_least_two_pilot_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        control_beta([(1.0, 1.0)])


def test_antithetic_sampler_consumes_exactly_n_from_inner_sampler() -> None:
    calls: list[int] = []

    def counting_sampler(n: int) -> list[float]:
        calls.append(n)
        return [0.5] * n

    wrapped = antithetic(counting_sampler, lambda u: 1.0 - u)
    pairs = list(wrapped(17))

    assert calls == [17]
    assert len(pairs) == 17
    assert all(pair == (0.5, 0.5) for pair in pairs)
