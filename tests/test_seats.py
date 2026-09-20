from __future__ import annotations

import hashlib
import random

from gamekit.seats import (
    _SEAT_RNG_PREFIX,
    rotate,
    seat_occupancy_counts,
    seat_rng,
    seat_rng_legacy,
)


def _seed_for(driver_seed: int, seat: int) -> int:
    payload = _SEAT_RNG_PREFIX + f"{driver_seed}:{seat}".encode()
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest, "big")


def test_seat_rng_legacy_pinned_reference_value() -> None:
    """Guards the frozen hash((driver_seed, seat)) contract: catan's
    pre-0.2.0 experiments/results/benchmark_*.json depend on this exact
    value. Verified byte-identical on CPython 3.13.3 and 3.14.5 -- if a
    future CPython release changes it, this test is where that breaks."""
    assert seat_rng_legacy(1, 0).random() == 0.258567530079989


def test_seat_rng_pinned_reference_values() -> None:
    """Pins both the intermediate seed integer and the first draw, for each
    of a few (driver_seed, seat) pairs -- pinning both makes a future
    failure diagnosable (the mix changed vs. random.Random itself changed)."""
    assert _seed_for(1, 0) == 173946671623960035630828504396598711115
    assert seat_rng(1, 0).random() == 0.34445139915349277
    assert seat_rng(7, 2).random() == 0.12280780436698058
    assert seat_rng(7, 3).random() == 0.5224307866652843


def test_seat_rng_is_a_random_random_keyed_by_seat_not_seat_order() -> None:
    a = seat_rng(driver_seed=7, seat=2)
    b = seat_rng(driver_seed=7, seat=2)
    assert isinstance(a, random.Random)
    assert a.random() == b.random()

    c = seat_rng(driver_seed=7, seat=3)
    assert seat_rng(driver_seed=7, seat=2).random() != c.random()


def test_seat_rng_is_injective_for_negative_and_wide_seeds() -> None:
    """The decimal ':'-joined encoding is injective over all ints, unlike a
    masked fixed-width mix -- assert the property, not just document it."""
    huge = 2**64 + 1
    assert seat_rng(-5, 0).random() != seat_rng(5, 0).random()
    assert seat_rng(huge, 0).random() != seat_rng(huge % 2**64, 0).random()
    assert seat_rng(1, -1).random() != seat_rng(-1, 1).random()


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
