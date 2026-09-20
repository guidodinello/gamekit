from __future__ import annotations

from collections.abc import Sequence

import pytest

from gamekit.benchmark import run_arm, summarize_by_role, summarize_by_seat


def test_summarize_by_seat_reports_gate_one_null_expectation() -> None:
    summary = summarize_by_seat([0, 1, 2, 3, 0, 1, 2, 3], num_seats=4)
    assert summary["0"]["wins"] == 2
    assert summary["0"]["n_games"] == 8
    assert summary["0"]["expected_under_null"] == 0.25


def test_summarize_by_role_aggregates_across_seats() -> None:
    role_lineups = [("a", "b"), ("b", "a")]
    winning_seats = [0, 0]  # seat 0 always wins: "a" then "b"
    summary = summarize_by_role(role_lineups, winning_seats)
    assert summary["a"]["n_seat_occupancies"] == 2
    assert summary["a"]["wins"] == 1
    assert summary["b"]["wins"] == 1
    assert summary["a"]["seat_occupancy_counts"] == {0: 1, 1: 1}


def _seat_zero_wins(pairs: Sequence[tuple[int, int]]) -> list[int]:
    return [0 for _ in pairs]


def test_run_arm_rejects_n_games_not_a_multiple_of_lineup_length() -> None:
    with pytest.raises(ValueError, match="multiple of the lineup length"):
        run_arm(
            lineup=("heuristic", "random", "random", "random"),
            num_seats=4,
            n_games=7,
            engine_seed_base=1,
            driver_seed_base=1,
            play=_seat_zero_wins,
            winning_seat=lambda r: r,
        )


def test_run_arm_computes_balanced_summaries_under_consecutive_seeds() -> None:
    lineup = ("heuristic", "random", "random", "random")
    payload = run_arm(
        lineup=lineup,
        num_seats=4,
        n_games=8,
        engine_seed_base=1,
        driver_seed_base=1,
        play=_seat_zero_wins,
        winning_seat=lambda r: r,
    )
    assert payload["by_role"]["heuristic"]["seat_occupancy_counts"] == {
        0: 2,
        1: 2,
        2: 2,
        3: 2,
    }
    # Role order in `comparisons`' key mirrors insertion order into by_role,
    # which depends on which role a game's rotated lineup puts in seat 0
    # first -- assert on the pair, not a specific ordering of the key.
    assert len(payload["comparisons"]) == 1
    (comparison_key,) = payload["comparisons"]
    assert set(comparison_key.split("_vs_")) == {"heuristic", "random"}
    assert payload["by_seat"]["0"]["wins"] == 8
    assert payload["n_games"] == 8
    assert payload["experiment"] == "benchmark"


def test_run_arm_raises_when_rotation_offset_is_constant() -> None:
    """If a caller passes a rotation_offset that doesn't advance across
    engine seeds, seat occupancy comes out unbalanced -- run_arm must catch
    this rather than silently returning a confounded result."""
    lineup = ("heuristic", "random", "random", "random")
    with pytest.raises(ValueError, match="unbalanced seat occupancy"):
        run_arm(
            lineup=lineup,
            num_seats=4,
            n_games=8,
            engine_seed_base=1,
            driver_seed_base=1,
            play=_seat_zero_wins,
            winning_seat=lambda r: r,
            rotation_offset=lambda engine_seed: 0,
        )
