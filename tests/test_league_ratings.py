"""Tests for gamekit.league.ratings: BT recovery, anchoring, bootstrap
coverage, separation, and cycle detection."""

from __future__ import annotations

import random

import pytest

from gamekit.league import (
    bootstrap_ratings,
    elo_ratings,
    find_cycles,
    fit_bradley_terry,
    win_matrix,
)

TRUE_ELO = {"random": 0.0, "weak": 100.0, "mid": 250.0, "strong": 400.0}
AGENTS = list(TRUE_ELO)


def _p(i: str, j: str) -> float:
    return 1 / (1 + 10 ** ((TRUE_ELO[j] - TRUE_ELO[i]) / 400))


def _expected_counts(n: int) -> dict[tuple[str, str], float]:
    return {(i, j): n * _p(i, j) for i in AGENTS for j in AGENTS if i < j} | {
        (j, i): n * _p(j, i) for i in AGENTS for j in AGENTS if i < j
    }


def _sampled_counts(n: int, rng: random.Random) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], float] = {}
    for i in AGENTS:
        for j in AGENTS:
            if i < j:
                w = rng.binomialvariate(n, _p(i, j))
                counts[(i, j)] = w
                counts[(j, i)] = n - w
    return counts


def test_bt_recovers_known_strengths_exactly_from_expected_counts() -> None:
    strengths = fit_bradley_terry(_expected_counts(100_000), AGENTS, prior_draws=0)
    elo = elo_ratings(strengths, "random")
    for agent, truth in TRUE_ELO.items():
        assert elo[agent] == pytest.approx(truth, abs=0.01)


def test_bt_recovers_known_strengths_from_sampled_games() -> None:
    counts = _sampled_counts(2000, random.Random(7))
    elo = elo_ratings(fit_bradley_terry(counts, AGENTS), "random")
    # At a ~0.91 win rate (strong vs random) the Elo standard error of one
    # pairing is ~13 even at n=2000, so 30 is about 3 sigma, not a loose fit.
    for agent, truth in TRUE_ELO.items():
        assert elo[agent] == pytest.approx(truth, abs=30)


def test_anchor_is_pinned_and_shifting_it_only_translates() -> None:
    strengths = fit_bradley_terry(_expected_counts(10_000), AGENTS)
    base = elo_ratings(strengths, "random")
    shifted = elo_ratings(strengths, "mid", anchor_rating=1000)
    assert base["random"] == 0
    assert shifted["mid"] == 1000
    offset = shifted["weak"] - base["weak"]
    for agent in AGENTS:
        assert shifted[agent] - base[agent] == pytest.approx(offset)


def test_unknown_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match="anchor"):
        elo_ratings({"a": 0.0, "b": 1.0}, "nope")


def test_bootstrap_anchor_interval_is_degenerate() -> None:
    counts = _sampled_counts(200, random.Random(1))
    cis = bootstrap_ratings(
        counts, AGENTS, anchor="weak", anchor_rating=50, n_bootstrap=50
    )
    assert tuple(cis["weak"]) == (50, 50)
    assert cis["strong"].lower < cis["strong"].upper


def test_bootstrap_intervals_cover_the_truth() -> None:
    covered = 0
    reps = 60
    for rep in range(reps):
        counts = _sampled_counts(200, random.Random(1000 + rep))
        cis = bootstrap_ratings(
            counts, AGENTS, anchor="random", n_bootstrap=100, seed=rep
        )
        ci = cis["mid"]
        covered += ci.lower <= TRUE_ELO["mid"] <= ci.upper
    assert covered / reps >= 0.85


def test_perfect_separation_stays_finite_with_the_prior() -> None:
    counts = {("a", "b"): 10.0, ("b", "a"): 0.0}
    elo = elo_ratings(fit_bradley_terry(counts, ["a", "b"]), "b")
    assert 0 < elo["a"] < 2000


def test_bootstrap_of_a_perfect_separation_is_not_falsely_certain() -> None:
    counts = {("a", "b"): 10.0, ("b", "a"): 0.0}
    cis = bootstrap_ratings(counts, ["a", "b"], anchor="b", n_bootstrap=200)
    assert cis["a"].width > 50  # not the zero-width interval a raw 10/10 gives


def test_plain_mle_rejects_perfect_separation_and_disconnection() -> None:
    with pytest.raises(ValueError, match="strongly connected"):
        fit_bradley_terry({("a", "b"): 10, ("b", "a"): 0}, ["a", "b"], prior_draws=0)
    disjoint = {("a", "b"): 5, ("b", "a"): 5, ("c", "d"): 5, ("d", "c"): 5}
    for prior in (0, 1.0):
        with pytest.raises(ValueError, match="connected"):
            fit_bradley_terry(disjoint, list("abcd"), prior_draws=prior)


def _rps_counts(edge: int, n: int = 100) -> dict[tuple[str, str], float]:
    return {
        ("rock", "scissors"): edge,
        ("scissors", "rock"): n - edge,
        ("scissors", "paper"): edge,
        ("paper", "scissors"): n - edge,
        ("paper", "rock"): edge,
        ("rock", "paper"): n - edge,
    }


def test_rock_paper_scissors_cycle_is_detected_though_ratings_tie() -> None:
    agents = ["rock", "paper", "scissors"]
    counts = _rps_counts(90)
    strengths = fit_bradley_terry(counts, agents)
    elo = elo_ratings(strengths, "rock")
    assert max(elo.values()) - min(elo.values()) < 1e-6  # the rating hides it
    matrix = win_matrix(counts, agents, strengths=strengths)
    assert find_cycles(matrix, agents) == [("paper", "rock", "scissors")]
    # ...and the matrix shows what the rating cannot: large BT residuals.
    assert abs(matrix[("rock", "scissors")]["residual"]) > 0.3


def test_transitive_matrix_and_noisy_edges_report_no_cycle() -> None:
    agents = ["a", "b", "c"]
    transitive = {
        ("a", "b"): 80, ("b", "a"): 20,
        ("b", "c"): 80, ("c", "b"): 20,
        ("a", "c"): 95, ("c", "a"): 5,
    }  # fmt: skip
    assert find_cycles(win_matrix(transitive, agents), agents) == []
    # A 52/100 edge's Wilson CI includes 0.5, so it is not an edge at all.
    noisy = _rps_counts(52)
    assert find_cycles(win_matrix(noisy, ["rock", "paper", "scissors"]), agents) == []
