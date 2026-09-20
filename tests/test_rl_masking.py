"""Tests for gamekit.rl.masking -- stdlib only, runs unconditionally in CI.
No numpy import here on purpose: see masking.py's module docstring for why
(gamekit#7, D5)."""

from __future__ import annotations

import random

from _fakegame import CLAIM, FakeCodec, FakeGame

from gamekit.rl.masking import coerce_to_legal, legal_action_mask


def test_legal_action_mask_reflects_main_phase() -> None:
    game = FakeGame(num_seats=3)
    state = game.reset()

    mask = legal_action_mask(game.legal_actions(state), FakeCodec())

    assert mask == [True, True, False]  # PASS, CLAIM legal; ACK not


def test_legal_action_mask_reflects_ack_phase() -> None:
    game = FakeGame(num_seats=3)
    state = game.reset()
    state = game.apply_action(state, CLAIM)

    mask = legal_action_mask(game.legal_actions(state), FakeCodec())

    assert mask == [False, False, True]


def test_legal_action_mask_is_all_false_when_nothing_is_legal() -> None:
    mask = legal_action_mask([], FakeCodec())

    assert mask == [False, False, False]


def test_coerce_to_legal_passes_through_an_already_legal_index() -> None:
    mask = [True, False, True]

    assert coerce_to_legal(0, mask, random.Random(0)) == 0
    assert coerce_to_legal(2, mask, random.Random(0)) == 2


def test_coerce_to_legal_replaces_an_illegal_index_with_a_legal_one() -> None:
    mask = [False, True, False]

    assert coerce_to_legal(0, mask, random.Random(0)) == 1


def test_coerce_to_legal_replaces_an_out_of_range_index() -> None:
    mask = [True, False]

    assert coerce_to_legal(99, mask, random.Random(0)) == 0
    assert coerce_to_legal(-1, mask, random.Random(0)) == 0
