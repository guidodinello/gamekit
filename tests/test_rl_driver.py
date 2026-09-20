"""Tests for gamekit.rl.driver.advance_until_learner -- stdlib only, runs
unconditionally in CI (see gamekit#7's D5: no gymnasium/numpy import here)."""

from __future__ import annotations

from _fakegame import FakeAgent, FakeGame, always_claim

from gamekit.rl.driver import advance_until_learner


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
