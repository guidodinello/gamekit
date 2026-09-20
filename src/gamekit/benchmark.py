"""Generic benchmark harness: run named agent-lineup arms and report win rates
with seat rotation.

Extracted from catan's ``experiments/benchmark.py``. A **mode registry** maps
a mode name to a role lineup (one role name per seat slot) -- this is the
mode-registry fix ``docs/shared-ml-package.md`` asked for, replacing
truco-py's 11 hardcoded ``--mode`` branches.

This module never names a game-record field. ``run_arm`` takes a ``play``
callable (run the games, return whatever result type the game wants) and a
``winning_seat`` accessor (pull the winning seat out of one result) -- the
game's own result type never crosses into this package. This is what keeps
this module usable by any game engine, catan's included, without the game
engine's record type becoming a gamekit dependency.

**Seat rotation is mandatory, not a refinement**, for the same reason it was
in catan: a heuristic-vs-random result is confounded unless role-to-seat
assignment rotates so each role occupies each seat an equal number of times
across an arm. See ``run_arm``'s docstring for the exact rotation contract.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

from gamekit.mc import two_proportion_test, wilson_interval
from gamekit.results import stamp
from gamekit.seats import rotate, seat_occupancy_counts

type ModeRegistry = dict[str, Callable[[int], tuple[str, ...]]]


def summarize_by_seat(
    winning_seats: Sequence[int | None], num_seats: int
) -> dict[str, Any]:
    """Per-seat win rate, regardless of which role occupied it.

    This is the gate-1 check every game needs (random vs. random -> 1/n per
    seat), since ``summarize_by_role`` collapses same-named roles across
    every seat they occupied.
    """
    n = len(winning_seats)
    summary: dict[str, Any] = {}
    for seat in range(num_seats):
        wins = sum(1 for w in winning_seats if w == seat)
        ci = wilson_interval(wins, n)
        summary[str(seat)] = {
            "n_games": n,
            "wins": wins,
            "win_rate": wins / n if n else None,
            "win_rate_wilson_ci": list(ci),
            "expected_under_null": 1 / num_seats if num_seats else None,
        }
    return summary


def summarize_by_role(
    role_lineups: Sequence[Sequence[str]], winning_seats: Sequence[int | None]
) -> dict[str, Any]:
    """Win rate per role name, aggregated across every seat the role
    occupied -- plus the rotation bookkeeping (role x seat counts) that
    proves the arm isn't confounded by an uneven seat assignment.

    ``role_lineups[i]`` is game ``i``'s rotated lineup (seat -> role name);
    ``winning_seats[i]`` is that game's winning seat, or ``None``.
    """
    by_role_wins: dict[str, int] = {}
    by_role_n: dict[str, int] = {}
    for lineup, winner in zip(role_lineups, winning_seats, strict=True):
        for seat, role in enumerate(lineup):
            by_role_n[role] = by_role_n.get(role, 0) + 1
            if winner == seat:
                by_role_wins[role] = by_role_wins.get(role, 0) + 1

    seat_counts = seat_occupancy_counts(role_lineups)
    summary: dict[str, Any] = {}
    for role, n in by_role_n.items():
        wins = by_role_wins.get(role, 0)
        ci = wilson_interval(wins, n)
        summary[role] = {
            "n_seat_occupancies": n,
            "wins": wins,
            "win_rate": wins / n,
            "win_rate_wilson_ci": list(ci),
            "seat_occupancy_counts": seat_counts.get(role, {}),
        }
    return summary


def run_arm[R](
    *,
    lineup: tuple[str, ...],
    num_seats: int,
    n_games: int,
    engine_seed_base: int,
    driver_seed_base: int,
    play: Callable[[Sequence[tuple[int, int]]], Sequence[R]],
    winning_seat: Callable[[R], int | None],
    rotation_offset: Callable[[int], int] = lambda engine_seed: engine_seed,
) -> dict[str, Any]:
    """Run ``n_games`` games of one mode's ``lineup``, rotating role-to-seat
    assignment, and report win rates by role and by seat.

    **Rotation contract**: for the rotation to be exact, an arm's engine
    seeds must be ``k`` consecutive integers, ``k = len(lineup)`` -- this
    is what ``(engine_seed_base, engine_seed_base + n_games - 1)`` gives
    you, and it is why ``n_games`` must be a multiple of ``k``. The
    rotation offset applied to each game is
    ``rotation_offset(engine_seed) %% len(lineup)`` (the modulo is applied
    here, not by the caller); the default ``rotation_offset`` is the
    identity, reproducing catan's ``engine_seed %% k`` convention with no
    argument needed.

    This contract is not just documented -- it is validated. After playing
    every game, this raises ``ValueError`` if any role's per-seat occupancy
    came out unbalanced (missing a seat, or an unequal count across seats),
    which would mean an arm's engine seeds silently were not ``k``
    consecutive integers and the result is confounded.

    ``play`` receives the full list of ``(engine_seed, driver_seed)`` pairs
    and returns one result per pair, in the same order; the game decides how
    those results get produced (sequentially, via a process pool, whatever).
    ``winning_seat`` reads the winning seat out of one result, or ``None``
    for a game with no winner.
    """
    if n_games % len(lineup) != 0:
        raise ValueError(
            f"n_games must be a multiple of the lineup length ({len(lineup)}) "
            "for exact seat rotation counts"
        )

    pairs = [(engine_seed_base + i, driver_seed_base + i) for i in range(n_games)]
    role_lineups = [
        rotate(lineup, rotation_offset(engine_seed) % len(lineup))
        for engine_seed, _ in pairs
    ]
    results = play(pairs)
    winning_seats = [winning_seat(r) for r in results]

    by_role = summarize_by_role(role_lineups, winning_seats)
    by_seat = summarize_by_seat(winning_seats, num_seats)

    for role, counts in seat_occupancy_counts(role_lineups).items():
        if len(counts) != num_seats or len(set(counts.values())) != 1:
            raise ValueError(
                f"role {role!r} has unbalanced seat occupancy {counts} -- "
                f"engine_seed_base..+n_games-1 must be exactly {len(lineup)} "
                "consecutive integers for exact rotation"
            )

    roles = list(by_role)
    comparisons: dict[str, Any] = {}
    if len(roles) == 2:
        a, b = roles
        test = two_proportion_test(
            by_role[a]["wins"],
            by_role[a]["n_seat_occupancies"],
            by_role[b]["wins"],
            by_role[b]["n_seat_occupancies"],
        )
        comparisons[f"{a}_vs_{b}"] = {"z": test.z, "p_value": test.p_value}

    return stamp(
        "benchmark",
        num_seats=num_seats,
        n_games=n_games,
        engine_seed_base=engine_seed_base,
        driver_seed_base=driver_seed_base,
        by_role=by_role,
        by_seat=by_seat,
        comparisons=comparisons,
    )


def build_parser(modes: ModeRegistry, prog: str) -> argparse.ArgumentParser:
    """The shared CLI skeleton every benchmark arm uses: ``--mode`` (from
    ``modes``), ``--games``, ``--players``, ``--engine-seed-base``,
    ``--driver-seed-base``, ``--workers``. The caller owns ``main()`` and any
    extra flags, added to the returned parser.
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--mode", choices=sorted(modes), required=True)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--engine-seed-base", type=int, default=1)
    parser.add_argument("--driver-seed-base", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    return parser
