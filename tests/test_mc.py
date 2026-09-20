"""Ported verbatim from catan's tests/test_experiments.py (mcstats section),
plus tests for the new Accumulator[V, S, R] protocol."""

from __future__ import annotations

import random
import statistics

import pytest

from gamekit.mc import (
    Accumulator,
    MeanAccumulator,
    benjamini_hochberg,
    sample_size_clt,
    sample_size_hoeffding,
    two_proportion_sample_size,
    two_proportion_test,
    wilson_interval,
)


def test_wilson_interval_matches_known_reference_values() -> None:
    ci = wilson_interval(50, 100)
    assert round(ci.lower, 3) == 0.404
    assert round(ci.upper, 3) == 0.596

    ci_extreme = wilson_interval(0, 20)
    assert 0.0 <= ci_extreme.lower < ci_extreme.upper < 1.0


def test_sample_size_formulas_match_the_plan_table() -> None:
    assert two_proportion_sample_size(1 / 3, 0.05) == 1396
    assert two_proportion_sample_size(1 / 3, 0.02) == 8721
    assert two_proportion_sample_size(1 / 3, 0.01) == 34884
    assert sample_size_clt(0.05, 0.05) == 385
    assert sample_size_hoeffding(0.05, 0.05) == 738
    assert sample_size_hoeffding(0.05, 0.05) >= sample_size_clt(0.05, 0.05)


def test_two_proportion_test_detects_no_difference_for_identical_arms() -> None:
    result = two_proportion_test(50, 100, 50, 100)
    assert result.z == 0.0
    assert result.p_value == 1.0


def test_benjamini_hochberg_flags_only_the_small_p_values() -> None:
    assert benjamini_hochberg([0.001, 0.2, 0.03, 0.5], q=0.05) == [
        True,
        False,
        False,
        False,
    ]
    assert benjamini_hochberg([]) == []


def test_mean_accumulator_matches_stdlib_statistics() -> None:
    rng = random.Random(42)
    samples = [rng.uniform(0, 1) for _ in range(500)]
    acc = MeanAccumulator[float](value=lambda x: x)
    acc.update_all(samples)

    assert round(acc.mean, 9) == round(statistics.mean(samples), 9)
    assert round(acc.variance, 6) == round(statistics.variance(samples), 6)

    ci = acc.result().confidence_interval()
    assert ci.lower < acc.mean < ci.upper


def test_mean_accumulator_projects_arbitrary_sample_types() -> None:
    """Fixes mmo-utils gap 2: this accumulator is not float-only at the
    call site -- the projection is supplied once at construction."""

    class Outcome:
        def __init__(self, score: int) -> None:
            self.score = score

    acc = MeanAccumulator[Outcome](value=lambda o: float(o.score))
    acc.update_all(Outcome(s) for s in (1, 2, 3, 4, 5))
    assert acc.mean == 3.0


class _SumBoth:
    """A plain, ordinary method-based ``Accumulator`` implementation --
    ``update``/``finalize`` are Protocol methods, so a class satisfies them
    the same way it satisfies any other Protocol method: no staticmethod
    wrapping, no ``type: ignore``. Folds a two-tuple ``V`` (unlike
    ``MeanAccumulator``, which can only ever produce a mean/variance)."""

    init = (0.0, 0.0)

    def update(
        self, state: tuple[float, float], value: tuple[float, float]
    ) -> tuple[float, float]:
        return (state[0] + value[0], state[1] + value[1])

    def finalize(self, state: tuple[float, float]) -> tuple[float, float]:
        return state


def test_accumulator_protocol_accepts_a_plain_method_based_implementation() -> None:
    """Fixes mmo-utils gap 2 fully: a custom accumulator can fold an
    arbitrary V (here, a two-tuple) into any state/result shape, which
    MeanAccumulator alone cannot express. A protocol that only a
    staticmethod-wrapped or `type: ignore`d class can satisfy would be a
    design bug -- this class is as ordinary as it gets, and this file being
    mypy-clean is the check."""
    acc: Accumulator[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        _SumBoth()
    )
    assert isinstance(acc, Accumulator)

    state = acc.init
    for value in [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]:
        state = acc.update(state, value)
    assert acc.finalize(state) == (6.0, 60.0)


@pytest.mark.parametrize("n", [0, 1])
def test_mean_accumulator_raises_below_minimum_sample_count(n: int) -> None:
    acc = MeanAccumulator[float](value=lambda x: x)
    acc.update_all(range(n))
    with pytest.raises(ZeroDivisionError):
        _ = acc.variance
