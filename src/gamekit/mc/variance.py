"""Variance reduction: antithetic and control variates.

Resolves mmo-utils' ``DESIGN.md`` gap 4. Neither ``monte_carlo`` nor
``monte_carlo_reduce`` had built-in variance reduction; mmo-utils'
``mc_integration.ipynb`` implemented antithetic variates inline, so that
logic wasn't reusable. Both techniques here are expressed purely as a
``Sampler``/``Evaluator`` wrapper, composing directly with
``gamekit.mc.sample`` -- no changes to ``monte_carlo``/``monte_carlo_reduce``
themselves were needed.

**Antithetic variates** (``antithetic`` + ``paired_mean``): draw ``n`` values
from an inner sampler, pair each with a caller-supplied reflection, and fold
the pair's *averaged* evaluation as one sample. If ``reflect`` induces
negative correlation between ``evaluator(x)`` and ``evaluator(reflect(x))``,
the paired-average estimator has strictly lower variance than plain MC at the
same number of underlying draws. ``reflect`` is necessarily
game/distribution-specific -- for ``X ~ U(0, 1)`` it is ``lambda u: 1.0 -
u``; for a symmetric two-team estimand (e.g. truco-py's
``flores_A >= 2 or flores_B >= 2``) it is a team swap. This is why
``antithetic`` takes ``reflect`` as a parameter rather than assuming
uniforms.

**The equal-budget comparison, precisely** (the thing every test in this
module's test file measures): calling
``monte_carlo(n, antithetic(sampler, reflect), paired_mean(evaluator))``
draws exactly ``n`` values from the *inner* sampler (one real draw per pair)
and evaluates ``2n`` points (``x`` and ``reflect(x)`` for each), folding them
into ``n`` paired-mean samples. Compare against plain
``monte_carlo(2 * n, sampler, evaluator)``, which also evaluates ``2n``
points but from ``2n`` independent draws. Both consume the same evaluation
budget; antithetic needs only half the independent randomness. The fair
comparison is the **variance of the estimator**, i.e. ``result.variance /
result.n`` (equivalently the confidence-interval half-width) on each side --
*not* a bare comparison of ``MCResult.variance``, since a paired-mean sample
has smaller spread than a raw sample regardless of correlation (halving a
sum's variance by averaging two terms), which would look like variance
reduction even with no effect at all. At equal estimator-variance
comparison, pairing with an **independent** second draw (zero correlation)
gives a ratio of exactly ~1 -- averaging alone buys nothing once you account
for using half as many paired samples; pairing with a self-reflection
(perfect positive correlation) makes the ratio ~0.5, i.e. *worse* than plain
MC; only genuine negative correlation moves the ratio above 1. The test
suite includes both negative controls specifically to prove the effect is
correlation, not the arithmetic of averaging.

**Control variates** (``control_variate`` + ``control_beta``): given a
control function ``c`` with a *known* ``E[c(X)]``, correlated with the
estimand ``f``, the adjusted evaluator ``f(x) - beta * (c(x) - E[c])`` is
still unbiased for ``E[f(X)]`` (since ``E[c(X) - E[c]] = 0``) but has lower
variance when ``beta`` is chosen well. ``control_beta`` estimates the
variance-minimizing ``beta = Cov(f, c) / Var(c)`` from a pilot sample of
``(f, c)`` pairs. Using the *same* draws for the pilot and for the final
estimate introduces a small bias (the pilot and the estimate are no longer
independent); the recommended usage is a separate pilot sample, stated here
rather than enforced, since a caller may have a domain reason to reuse draws.

**Importance sampling** (mmo-utils gap 4's "stretch" item, and the task's
"only if it falls out naturally") needs no new surface under this design: to
estimate ``E_p[f(X)]`` by sampling from a proposal ``q`` instead of ``p``,
draw from ``q`` and evaluate the weighted quantity directly --

    monte_carlo(n, sampler_from_q, lambda x: f(x) * p_density(x) / q_density(x))

-- which is exactly ``monte_carlo``'s existing ``(sampler, evaluator)``
shape. No wrapper is added because there is nothing generic left to
factor out: the importance weight is inseparable from the specific ``p``/``q``
pair a caller has in mind.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable

from gamekit.mc.sample import Evaluator, Sampler


def antithetic[T](
    sampler: Sampler[T], reflect: Callable[[T], T]
) -> Sampler[tuple[T, T]]:
    """Wrap ``sampler`` so each draw ``x`` is paired with ``reflect(x)``.
    Consumes exactly ``n`` values from the inner ``sampler`` per call, one
    real draw per pair -- ``reflect`` costs no additional randomness. Pair
    with ``paired_mean`` to fold each pair into one averaged evaluation.
    """

    def sample(n: int) -> Iterable[tuple[T, T]]:
        return ((x, reflect(x)) for x in sampler(n))

    return sample


def paired_mean[T](evaluator: Evaluator[T, float]) -> Evaluator[tuple[T, T], float]:
    """Fold an antithetic pair into the average of its two evaluations --
    the sample that ``monte_carlo``/``monte_carlo_reduce`` then accumulates.
    """

    def evaluate(pair: tuple[T, T]) -> float:
        x, y = pair
        return (evaluator(x) + evaluator(y)) / 2.0

    return evaluate


def control_variate[T](
    evaluator: Evaluator[T, float],
    control: Evaluator[T, float],
    control_mean: float,
    beta: float,
) -> Evaluator[T, float]:
    """Adjust ``evaluator`` by a control variate: ``f(x) - beta * (c(x) -
    E[c])``. Still unbiased for ``E[f(X)]`` for any ``beta`` (since
    ``E[c(X) - control_mean] = 0`` when ``control_mean`` is genuinely
    ``E[c(X)]``); ``beta = 0`` reproduces ``evaluator`` exactly. Use
    ``control_beta`` to estimate the variance-minimizing ``beta`` from a
    pilot sample, or supply a known analytic value directly.
    """

    def evaluate(x: T) -> float:
        return evaluator(x) - beta * (control(x) - control_mean)

    return evaluate


def control_beta(pilot: Iterable[tuple[float, float]]) -> float:
    """Estimate the variance-minimizing control-variate coefficient,
    ``beta = Cov(f, c) / Var(c)``, from a pilot sample of ``(f(x), c(x))``
    pairs. Prefer a pilot sample drawn separately from the final estimate --
    reusing the same draws for both introduces a small bias.
    """
    values = list(pilot)
    if len(values) < 2:
        raise ValueError("need at least 2 pilot samples to estimate beta")
    f_values = [f for f, _ in values]
    c_values = [c for _, c in values]
    control_variance = statistics.variance(c_values)
    if control_variance == 0:
        raise ValueError("control has zero variance in the pilot sample")
    return statistics.covariance(f_values, c_values) / control_variance
