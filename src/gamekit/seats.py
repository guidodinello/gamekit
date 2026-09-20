"""Seat rotation and per-seat RNG streams.

Extracted from catan's ``experiments/benchmark.py`` and ``server/bots.py``,
which had converged on the same idiom independently in three places
(``benchmark.py``, ``server/bots.py``, and inline in two test factories).

Both functions are keyed by *seat* (a slot in the turn order), not by player id
or agent index -- swapping the agent in one seat never perturbs another seat's
draws or its role assignment, which is what makes an arm's seat rotation exact.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence


def seat_rng(driver_seed: int, seat: int) -> random.Random:
    """A ``random.Random`` stream owned by one seat.

    Frozen deliberately as ``random.Random(hash((driver_seed, seat)))`` --
    this exact expression, not a documented-stable replacement. ``hash()`` of
    an int tuple is an unspecified CPython implementation detail, but it has
    been verified byte-identical across CPython 3.13 and 3.14 (the only two
    versions this package supports), and changing it would invalidate every
    already-committed benchmark result that depends on this stream. See
    ``tests/test_seats.py`` for the pinned reference values that guard this
    contract -- if a future CPython version breaks them, that test is where it
    will be caught, not silently in a downstream game's numbers.
    """
    return random.Random(hash((driver_seed, seat)))


def rotate[T](lineup: Sequence[T], offset: int) -> tuple[T, ...]:
    """Rotate ``lineup`` left by ``offset`` (mod ``len(lineup)``).

    ``offset`` is taken as given -- the caller decides what it means (e.g. an
    ``engine_seed`` directly, or ``engine_seed %% k`` already applied). This
    function only rotates; it does not interpret seeds.
    """
    if not lineup:
        return tuple(lineup)
    r = offset % len(lineup)
    return tuple(lineup[r:]) + tuple(lineup[:r])


def seat_occupancy_counts(
    lineups: Iterable[Sequence[str]],
) -> dict[str, dict[int, int]]:
    """Per-role, per-seat occupancy counts across a sequence of rotated lineups.

    This is the rotation-fairness proof: for an arm to be unconfounded, each
    role must occupy each seat an equal number of times. Returns
    ``{role: {seat: count}}``, seats sorted ascending.
    """
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for lineup in lineups:
        for seat, role in enumerate(lineup):
            counts[role][seat] += 1
    return {role: dict(sorted(seats.items())) for role, seats in counts.items()}
