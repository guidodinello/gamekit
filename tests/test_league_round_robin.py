"""Tests for gamekit.league.round_robin: scheduling, resume, roster
independence, and a tiny end-to-end league with stub agents."""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from gamekit.league import (
    ScheduledGame,
    load_pairings,
    pairing_lineup,
    pairing_seeds,
    run_league,
    run_pairing,
    summarize_league,
)
from gamekit.seats import seat_rng

STRENGTH = {"random": 1.0, "mid": 2.0, "strong": 4.0, "twin_a": 3.0, "twin_b": 3.0}
SEAT0_BONUS = 1.5  # a positional advantage that rotation must cancel


class StubPlayer:
    """Plays a Luce-model game among the agents seated, counting calls and
    optionally failing on the n-th pairing to simulate an interrupted run."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, games: Sequence[ScheduledGame]) -> list[int]:
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("simulated crash")
        out = []
        for game in games:
            weights = [
                STRENGTH[a] * (SEAT0_BONUS if seat == 0 else 1.0)
                for seat, a in enumerate(game.lineup)
            ]
            draw = seat_rng(game.driver_seed, 0).random() * sum(weights)
            winner = len(weights) - 1
            for seat, w in enumerate(weights):
                draw -= w
                if draw < 0:
                    winner = seat
                    break
            out.append(winner)
        return out


def _league(
    tmp_path: Path,
    player: StubPlayer,
    agents: Sequence[str] = ("random", "mid", "strong"),
    num_seats: int = 2,
    n: int = 400,
    name: str = "L",
) -> dict[str, Any]:
    return run_league(
        agents=agents,
        num_seats=num_seats,
        n_per_pairing=n,
        seed=11,
        play=player,
        winning_seat=lambda seat: seat,
        results_dir=tmp_path,
        name=name,
        anchor="random",
        n_bootstrap=50,
    )


def _strip(obj: Any) -> Any:
    """Drop the run-time stamps that legitimately differ between runs."""
    if isinstance(obj, dict):
        return {
            k: _strip(v)
            for k, v in obj.items()
            if k not in {"generated_at", "git_commit"}
        }
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def test_end_to_end_two_seat_league_recovers_the_strength_order(
    tmp_path: Path,
) -> None:
    summary = _league(tmp_path, StubPlayer())
    ratings = summary["ratings"]
    assert list(ratings) == ["strong", "mid", "random"]
    assert ratings["random"]["elo"] == 0
    assert ratings["random"]["elo_ci"] == [0, 0]
    # Luce strengths 1:2:4 -> true gaps of 400*log10(2) = 120 and 240 Elo.
    assert 80 < ratings["mid"]["elo"] < 160
    assert 190 < ratings["strong"]["elo"] < 290
    assert (tmp_path / "L" / "league.json").exists()
    assert summary["cycles"] == []
    # Seat rotation is validated by run_arm; the matrix carries Wilson CIs.
    cell = summary["matrix"]["strong__vs__random"]
    assert (
        cell["win_rate_wilson_ci"][0] < cell["win_rate"] < cell["win_rate_wilson_ci"][1]
    )


def test_identical_agents_in_a_four_seat_game_rate_equal(tmp_path: Path) -> None:
    summary = _league(
        tmp_path, StubPlayer(), agents=("random", "twin_a", "twin_b"), num_seats=4,
        n=800,
    )  # fmt: skip
    ratings = summary["ratings"]
    assert abs(ratings["twin_a"]["elo"] - ratings["twin_b"]["elo"]) < 50


def test_odd_seat_count_is_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="evenly"):
        _league(tmp_path, StubPlayer(), num_seats=3, n=3)
    assert pairing_lineup("a", "b", 4) == ("a", "b", "a", "b")
    with pytest.raises(ValueError):
        pairing_lineup("a", "b", 1)


def test_resume_reproduces_an_uninterrupted_run(tmp_path: Path) -> None:
    full_player = StubPlayer()
    full = _league(tmp_path, full_player, name="full")

    crashing = StubPlayer(fail_on_call=3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _league(tmp_path, crashing, name="resumed")
    assert len(load_pairings(tmp_path / "resumed")) == 2  # the 3rd never landed

    resumer = StubPlayer()
    resumed = _league(tmp_path, resumer, name="resumed")
    assert resumer.calls == 1  # only the missing pairing was played

    assert _strip(resumed) == _strip(full)
    for a, b in (("mid", "random"), ("mid", "strong"), ("random", "strong")):
        stem = f"{a}__vs__{b}.json"
        assert _strip(json.loads((tmp_path / "resumed" / stem).read_text())) == _strip(
            json.loads((tmp_path / "full" / stem).read_text())
        )


def test_resume_refuses_a_changed_config(tmp_path: Path) -> None:
    _league(tmp_path, StubPlayer())
    with pytest.raises(ValueError, match="different config"):
        _league(tmp_path, StubPlayer(), n=800)


def test_adding_an_agent_leaves_existing_pairings_untouched(tmp_path: Path) -> None:
    _league(tmp_path, StubPlayer())
    before = {p.name: p.read_text() for p in (tmp_path / "L").glob("*__vs__*.json")}
    player = StubPlayer()
    _league(tmp_path, player, agents=("random", "mid", "strong", "twin_a"))
    assert player.calls == 3  # only the three pairings involving twin_a
    for fname, text in before.items():
        assert (tmp_path / "L" / fname).read_text() == text
    assert pairing_seeds(11, "a", "b") == pairing_seeds(11, "b", "a")
    assert pairing_seeds(11, "a", "b") != pairing_seeds(12, "a", "b")


def test_shrinking_the_roster_drops_the_removed_agent_from_the_summary(
    tmp_path: Path,
) -> None:
    _league(tmp_path, StubPlayer(), agents=("random", "mid", "strong", "twin_a"))
    player = StubPlayer()
    summary = _league(tmp_path, player)  # same dir, three agents
    assert player.calls == 0  # every needed pairing is already on disk
    assert summary["agents"] == ["mid", "random", "strong"]
    assert not any("twin_a" in key for key in summary["matrix"])


def test_agent_names_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        _league(tmp_path, StubPlayer(), agents=("random", "random"))
    with pytest.raises(ValueError, match="may not contain"):
        _league(tmp_path, StubPlayer(), agents=("random", "a__vs__b"))


def _pairing(tmp_path: Path, player: StubPlayer, a: str, b: str, **kw: Any) -> Any:
    return run_pairing(
        a, b, num_seats=2, n_per_pairing=400, seed=11, play=player,
        winning_seat=lambda seat: seat, results_dir=tmp_path, name="L", **kw,
    )  # fmt: skip


def test_config_hash_is_pinned_so_existing_result_dirs_keep_resuming(
    tmp_path: Path,
) -> None:
    # Computed on 0.4.0. Changing it orphans every committed league directory.
    payload = run_pairing(
        "a", "b", num_seats=4, n_per_pairing=40, seed=11, play=StubPlayerAB(),
        winning_seat=lambda seat: seat, results_dir=tmp_path, name="L",
    )  # fmt: skip
    assert payload["config_hash"] == "70c17f89ebde"
    assert (tmp_path / "L" / "a__vs__b.json").exists()


class StubPlayerAB(StubPlayer):
    def __call__(self, games: Sequence[ScheduledGame]) -> list[int]:
        return [0] * len(games)


def test_run_pairing_writes_what_run_league_writes_and_no_summary(
    tmp_path: Path,
) -> None:
    full = _league(tmp_path, StubPlayer(), name="L")
    pairs = (("mid", "random"), ("mid", "strong"), ("random", "strong"))
    other = tmp_path / "P"
    for a, b in pairs:  # reversed on purpose: order-insensitive
        run_pairing(
            b, a, num_seats=2, n_per_pairing=400, seed=11, play=StubPlayer(),
            winning_seat=lambda seat: seat, results_dir=other.parent, name="P",
        )  # fmt: skip
    assert not (other / "league.json").exists()
    for a, b in pairs:
        stem = f"{a}__vs__{b}.json"
        assert _strip(json.loads((other / stem).read_text())) == _strip(
            json.loads((tmp_path / "L" / stem).read_text())
        )
    resummed = summarize_league(load_pairings(other), anchor="random", n_bootstrap=50)
    assert _strip(resummed) == _strip(full)


def test_run_pairing_resumes_without_playing_and_refuses_a_changed_config(
    tmp_path: Path,
) -> None:
    first = _pairing(tmp_path, StubPlayer(), "mid", "strong")
    player = StubPlayer()
    again = _pairing(tmp_path, player, "strong", "mid")
    assert player.calls == 0
    # `first` is the in-memory payload (tuples); `again` came back through JSON.
    assert again == json.loads(json.dumps(first))
    with pytest.raises(ValueError, match="different config"):
        run_pairing(
            "mid", "strong", num_seats=2, n_per_pairing=800, seed=11,
            play=StubPlayer(), winning_seat=lambda seat: seat,
            results_dir=tmp_path, name="L",
        )  # fmt: skip


def test_run_pairing_validates_its_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        _pairing(tmp_path, StubPlayer(), "mid", "mid")
    with pytest.raises(ValueError, match="may not contain"):
        _pairing(tmp_path, StubPlayer(), "mid", "a__vs__b")
    with pytest.raises(NotImplementedError, match="evenly"):
        run_pairing(
            "a", "b", num_seats=3, n_per_pairing=3, seed=11, play=StubPlayer(),
            winning_seat=lambda seat: seat, results_dir=tmp_path, name="L",
        )  # fmt: skip


def test_concurrent_pairings_land_the_same_files_as_a_serial_league(
    tmp_path: Path,
) -> None:
    _league(tmp_path, StubPlayer(), name="serial")
    pairs = (("mid", "random"), ("mid", "strong"), ("random", "strong"))
    with ThreadPoolExecutor(3) as pool:
        list(
            pool.map(
                lambda p: run_pairing(
                    *p, num_seats=2, n_per_pairing=400, seed=11,
                    play=StubPlayer(), winning_seat=lambda seat: seat,
                    results_dir=tmp_path, name="par",
                ),
                pairs,
            )
        )  # fmt: skip
    for a, b in pairs:
        stem = f"{a}__vs__{b}.json"
        assert _strip(json.loads((tmp_path / "par" / stem).read_text())) == _strip(
            json.loads((tmp_path / "serial" / stem).read_text())
        )
