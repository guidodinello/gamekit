"""Tests for gamekit.rl.driver.advance_until_learner -- stdlib only, runs
unconditionally in CI (see gamekit#7's D5: no gymnasium/numpy import here)."""

from __future__ import annotations

import random

import pytest
from _fakegame import FakeAgent, FakeGame, ScriptedDealGame, always_claim

from gamekit.rl.driver import (
    advance_until_learner,
    deal_until_learner,
    place_fixed_agents,
)


def test_stops_immediately_when_the_learner_already_owes_the_decision() -> None:
    game = FakeGame(num_seats=3, step_limit=100)
    state = game.reset()  # turn_player starts at seat 0
    agents = [None, FakeAgent("b"), FakeAgent("c")]

    steps = advance_until_learner(game, state, agents)

    assert steps == 0
    assert game.acting_player(state) == 0


def test_steps_every_opponent_seat_until_the_learner_must_act() -> None:
    game = FakeGame(num_seats=3, step_limit=100)
    state = game.reset()
    state.turn_player = 1  # start away from the learner's seat
    b, c = FakeAgent("b"), FakeAgent("c")
    agents = [None, b, c]

    steps = advance_until_learner(game, state, agents)

    # b (seat 1) and c (seat 2) each PASS once (their default policy is
    # "take the first legal action", which is PASS), landing back on the
    # learner's seat 0.
    assert steps == 2
    assert game.acting_player(state) == 0
    assert b.choose_calls == 1
    assert c.choose_calls == 1


def test_learner_is_woken_for_an_out_of_turn_ack_owed_mid_opponent_turn() -> None:
    """An opponent's CLAIM queues every *other* seat for an ACK, including
    the learner -- the learner must get control there too, not only on its
    nominal MAIN turn (gamekit#7, D3a: the seam is acting_player, not
    current_player)."""
    game = FakeGame(num_seats=3, step_limit=100)
    state = game.reset()
    state.turn_player = 1
    agents = [None, FakeAgent("b", always_claim), FakeAgent("c")]

    steps = advance_until_learner(game, state, agents)

    # b (seat 1) claims once; that single action queues seats 0 (the
    # learner) and 2 for ACK, and the learner is first in that queue --
    # so the driver stops after exactly one step, on an ACK decision, not
    # a MAIN one.
    assert steps == 1
    assert game.acting_player(state) == 0
    assert state.pending == [0, 2]
    assert state.scores[1] == 1


def test_stops_when_the_game_ends_before_the_learner_ever_acts() -> None:
    game = FakeGame(num_seats=3, step_limit=1)
    state = game.reset()
    state.turn_player = 1
    agents = [None, FakeAgent("b"), FakeAgent("c")]

    steps = advance_until_learner(game, state, agents)

    assert steps == 1
    assert game.is_terminal(state)


@pytest.mark.parametrize("learner_seat", [0, 1, 2])
def test_place_fixed_agents_fills_non_learner_seats_in_order(
    learner_seat: int,
) -> None:
    b, c = FakeAgent("b"), FakeAgent("c")

    placed = place_fixed_agents([b, c], 3, learner_seat)

    assert placed[learner_seat] is None
    assert [a for a in placed if a is not None] == [b, c]


def test_place_fixed_agents_rejects_a_wrong_agent_count() -> None:
    with pytest.raises(ValueError, match="num_seats - 1"):
        place_fixed_agents([FakeAgent("b")], 3, 0)
    with pytest.raises(ValueError, match="num_seats - 1"):
        place_fixed_agents([FakeAgent("a"), FakeAgent("b"), FakeAgent("c")], 3, 0)


@pytest.mark.parametrize("learner_seat", [-1, 3])
def test_place_fixed_agents_rejects_a_learner_seat_out_of_range(
    learner_seat: int,
) -> None:
    with pytest.raises(ValueError, match="learner_seat"):
        place_fixed_agents([FakeAgent("b"), FakeAgent("c")], 3, learner_seat)


def test_deal_until_learner_returns_a_state_where_the_learner_acts() -> None:
    game = ScriptedDealGame(step_limit=100)
    agents = [None, FakeAgent("b"), FakeAgent("c")]

    state = deal_until_learner(game, random.Random(0), agents)

    assert not game.is_terminal(state)
    assert game.acting_player(state) == 0
    assert len(game.reset_seeds) == 1


def test_deal_until_learner_redeals_and_calls_on_deal_per_attempt() -> None:
    game = ScriptedDealGame(step_limit=100, terminal_deals=2)
    agents = [None, FakeAgent("b"), FakeAgent("c")]
    dealt: list[object] = []

    state = deal_until_learner(game, random.Random(0), agents, on_deal=dealt.append)

    assert not game.is_terminal(state)
    assert len(dealt) == 3  # two discarded deals + the kept one
    assert dealt[-1] is state


def test_deal_until_learner_draws_one_seed_per_attempt_from_rng() -> None:
    game = ScriptedDealGame(step_limit=100, terminal_deals=2)
    agents = [None, FakeAgent("b"), FakeAgent("c")]

    deal_until_learner(game, random.Random(7), agents)

    rng = random.Random(7)
    assert game.reset_seeds == [rng.randint(0, 2**31 - 1) for _ in range(3)]


def test_deal_until_learner_raises_after_exactly_max_redeals() -> None:
    game = ScriptedDealGame(step_limit=100, terminal_deals=10**6)
    agents = [None, FakeAgent("b"), FakeAgent("c")]

    with pytest.raises(RuntimeError, match="3 deals"):
        deal_until_learner(game, random.Random(0), agents, max_redeals=3)

    assert len(game.reset_seeds) == 3


def test_deal_until_learner_succeeds_on_the_last_allowed_attempt() -> None:
    game = ScriptedDealGame(step_limit=100, terminal_deals=2)
    agents = [None, FakeAgent("b"), FakeAgent("c")]

    state = deal_until_learner(game, random.Random(0), agents, max_redeals=3)

    assert not game.is_terminal(state)


def test_deal_until_learner_rejects_max_redeals_below_one() -> None:
    game = ScriptedDealGame()

    with pytest.raises(ValueError, match="max_redeals"):
        deal_until_learner(game, random.Random(0), [None], max_redeals=0)
