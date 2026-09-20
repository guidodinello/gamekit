from __future__ import annotations

import random

from gamekit.seats import rotate, seat_occupancy_counts, seat_rng


def test_seat_rng_pinned_reference_value() -> None:
    """Guards the frozen hash((driver_seed, seat)) contract: catan's
    committed experiments/results/benchmark_*.json depend on this exact
    value. Verified byte-identical on CPython 3.13.3 and 3.14.5 -- if a
    future CPython release changes it, this test is where that breaks."""
    assert seat_rng(1, 0).random() == 0.258567530079989


def test_seat_rng_is_a_random_random_keyed_by_seat_not_seat_order() -> None:
    a = seat_rng(driver_seed=7, seat=2)
    b = seat_rng(driver_seed=7, seat=2)
    assert isinstance(a, random.Random)
    assert a.random() == b.random()

    c = seat_rng(driver_seed=7, seat=3)
    assert seat_rng(driver_seed=7, seat=2).random() != c.random()


def test_rotate_shifts_by_offset_mod_length() -> None:
    lineup = ("a", "b", "c", "d")
    assert rotate(lineup, 0) == lineup
    assert rotate(lineup, 1) == ("b", "c", "d", "a")
    assert rotate(lineup, 4) == lineup  # wraps
    assert rotate(lineup, 5) == ("b", "c", "d", "a")


def test_rotate_empty_lineup() -> None:
    assert rotate((), 3) == ()


def test_seat_occupancy_counts_is_balanced_under_consecutive_rotation() -> None:
    lineup = ("heuristic", "random", "random", "random")
    k = len(lineup)
    lineups = [rotate(lineup, seed % k) for seed in range(1, 1 + 4 * k)]
    counts = seat_occupancy_counts(lineups)

    assert set(counts["heuristic"].values()) == {4}
    assert set(counts["random"].values()) == {12}
    assert list(counts["heuristic"]) == [0, 1, 2, 3]
