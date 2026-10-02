"""League ratings: round-robin cross-play with anchored Bradley-Terry/Elo
ratings, the raw pairwise win matrix, and a non-transitivity report.

Stdlib-only, additive, and generic over games the same way
``gamekit.benchmark`` is: ``run_league`` takes a ``play`` callable and a
``winning_seat`` accessor and never names a game-record field. See
``docs/research/022-league-ratings.md`` for the design and its evidence, and
``round_robin``'s module docstring for the multi-player reduction.

A single rating hides rock-paper-scissors structure, so ``summarize_league``
always returns the win matrix and the cycle report next to the ratings.
"""

from __future__ import annotations

from gamekit.league.ratings import (
    PairwiseCounts,
    bootstrap_ratings,
    elo_ratings,
    find_cycles,
    fit_bradley_terry,
    predicted_win_rate,
    win_matrix,
)
from gamekit.league.round_robin import (
    ScheduledGame,
    load_pairings,
    pairing_lineup,
    pairing_seeds,
    run_league,
    summarize_league,
)

__all__ = [
    "PairwiseCounts",
    "ScheduledGame",
    "bootstrap_ratings",
    "elo_ratings",
    "find_cycles",
    "fit_bradley_terry",
    "load_pairings",
    "pairing_lineup",
    "pairing_seeds",
    "predicted_win_rate",
    "run_league",
    "summarize_league",
    "win_matrix",
]
