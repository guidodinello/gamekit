"""Seat rotation and per-seat RNG streams.

Extracted from catan's ``experiments/benchmark.py`` and ``server/bots.py``,
which had converged on the same idiom independently in three places
(``benchmark.py``, ``server/bots.py``, and inline in two test factories).

Both functions are keyed by *seat* (a slot in the turn order), not by player id
or agent index -- swapping the agent in one seat never perturbs another seat's
draws or its role assignment, which is what makes an arm's seat rotation exact.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

_SEAT_RNG_PREFIX = b"gamekit.seats.seat_rng.v2\x00"


def seat_rng(driver_seed: int, seat: int) -> random.Random:
    """A ``random.Random`` stream owned by one seat.

    Seeded from a BLAKE2b digest of ``driver_seed`` and ``seat``, not
    ``hash()`` -- BLAKE2b is specified by RFC 7693 with published test
    vectors, so ``hashlib.blake2b`` cannot change its output across CPython
    versions without breaking the standard. This mix is stable **by
    construction**, unlike the ``hash()``-based stream it replaces (still
    available as ``seat_rng_legacy``), which was an unspecified CPython
    implementation detail.

    ``driver_seed`` and ``seat`` are rendered in decimal (a leading ``-`` for
    negatives) and joined with ``:``, a character that cannot occur in a
    decimal integer -- so the encoding is injective over *all* ints, with no
    width truncation (unlike, say, masking into a fixed-width word). The
    module-qualified prefix domain-separates this stream from any other
    digest-seeded RNG gamekit might add later.
    """
    payload = _SEAT_RNG_PREFIX + f"{driver_seed}:{seat}".encode()
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return random.Random(int.from_bytes(digest, "big"))


def seat_rng_legacy(driver_seed: int, seat: int) -> random.Random:
    """The pre-0.2.0 ``seat_rng`` stream, kept only to replay results
    committed before the migration to :func:`seat_rng`.

    Frozen deliberately as ``random.Random(hash((driver_seed, seat)))`` --
    this exact expression, not a documented-stable replacement. ``hash()`` of
    an int tuple is an unspecified CPython implementation detail, but it has
    been verified byte-identical across CPython 3.13 and 3.14 (the only two
    versions this package supports). See ``tests/test_seats.py`` for the
    pinned reference value that guards this contract -- if a future CPython
    version breaks it, that test is where it will be caught, not silently in
    a downstream game's numbers.

    Deprecated: use :func:`seat_rng` for anything new. This function will not
    be removed while any committed result still depends on it.
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
