"""Tests for gamekit.rl.env.SingleAgentEnv against the fake toy game.

Requires gymnasium + numpy (the ``[rl]`` extra); skips cleanly at collection
when they're absent, so default CI (no extras installed) never fails here --
see gamekit#7's D5.
"""

from __future__ import annotations

import random
from typing import Any, Literal

import pytest

pytest.importorskip("gymnasium")

import numpy as np
from _fakegame import ACK, CLAIM, PASS, FakeAgent, FakeCodec, FakeGame, always_claim
from gymnasium import spaces

from gamekit.rl.env import SingleAgentEnv
from gamekit.rl.selfplay import OpponentPool

_OBS_SPACE = spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32)


def _encode(state: Any, seat: int) -> np.ndarray:
    return np.array([state.scores[seat], state.steps], dtype=np.float32)


class _ScoreReward:
    """Terminal reward = the learner's own score; 0 mid-episode."""

    def compute(self, state: Any, seat: int, done: bool) -> float:
        return float(state.scores[seat]) if done else 0.0

    def on_episode_start(self, state: Any, seat: int) -> None:
        pass


def _make_env(
    *,
    step_limit: int = 100,
    on_illegal: Literal["coerce", "raise"] = "coerce",
    opponent_policy: Any = None,
) -> SingleAgentEnv:
    game = FakeGame(num_seats=3, step_limit=step_limit)
    agents = [FakeAgent("b", opponent_policy), FakeAgent("c", opponent_policy)]
    return SingleAgentEnv(
        game=game,
        codec=FakeCodec(),
        encode=_encode,
        reward=_ScoreReward(),
        observation_space=_OBS_SPACE,
        num_seats=3,
        agents=agents,
        randomize_seat=False,
        on_illegal=on_illegal,
        seed=0,
    )


def test_constructor_requires_exactly_one_of_agents_or_opponent_pool() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SingleAgentEnv(
            game=FakeGame(num_seats=3),
            codec=FakeCodec(),
            encode=_encode,
            reward=_ScoreReward(),
            observation_space=_OBS_SPACE,
            num_seats=3,
        )


def test_constructor_rejects_both_agents_and_opponent_pool() -> None:
    pool: OpponentPool = OpponentPool(
        ".",
        "*.zip",
        load_opponent=lambda path: FakeAgent("x"),
        baseline_factory=lambda: FakeAgent("baseline"),
    )
    with pytest.raises(ValueError, match="exactly one"):
        SingleAgentEnv(
            game=FakeGame(num_seats=3),
            codec=FakeCodec(),
            encode=_encode,
            reward=_ScoreReward(),
            observation_space=_OBS_SPACE,
            num_seats=3,
            agents=[FakeAgent("b"), FakeAgent("c")],
            opponent_pool=pool,
        )


def test_constructor_validates_fixed_agent_count() -> None:
    with pytest.raises(ValueError, match="num_seats - 1"):
        SingleAgentEnv(
            game=FakeGame(num_seats=3),
            codec=FakeCodec(),
            encode=_encode,
            reward=_ScoreReward(),
            observation_space=_OBS_SPACE,
            num_seats=3,
            agents=[FakeAgent("b")],
        )


def test_reset_returns_an_observation_shaped_for_the_encoder() -> None:
    env = _make_env()

    obs, info = env.reset(seed=1)

    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert info == {}


def test_reset_places_the_learner_at_seat_zero_when_randomize_seat_is_false() -> None:
    # With step_limit=1, one CLAIM by the learner ends the episode, and the
    # terminal reward is the learner's own score -- so a reward of 1.0
    # proves the CLAIM was credited to seat 0, i.e. the learner.
    env = _make_env(step_limit=1)
    env.reset(seed=1)

    _, reward, terminated, _, _ = env.step(np.int64(CLAIM))

    assert terminated
    assert reward == 1.0


def test_action_masks_reflect_the_main_phase_after_reset() -> None:
    env = _make_env()
    env.reset(seed=1)

    mask = env.action_masks()

    assert mask.dtype == np.bool_
    assert list(mask) == [True, True, False]


def test_step_advances_opponents_until_the_learner_must_act_again() -> None:
    env = _make_env()
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(np.int64(PASS))

    assert not terminated
    assert not truncated
    assert obs.shape == (2,)
    # Opponents (seats 1, 2) default to PASS too, so control returns to the
    # learner with no claims/acks having fired.
    assert list(env.action_masks()) == [True, True, False]


def test_learner_is_woken_for_an_out_of_turn_ack_phase() -> None:
    """An opponent's CLAIM queues every other seat -- including the learner
    -- for an ACK; the env must surface that as the learner's next decision,
    not skip past it (gamekit#7, D3a)."""
    env = _make_env(opponent_policy=always_claim)
    env.reset(seed=1)
    env.step(np.int64(PASS))  # learner passes, handing the turn to seat 1

    # Seat 1's policy always claims, which queues seats 0 (learner) and 2.
    assert list(env.action_masks()) == [False, False, True]


def test_step_raises_on_illegal_action_when_on_illegal_is_raise() -> None:
    env = _make_env(on_illegal="raise")
    env.reset(seed=1)

    with pytest.raises(ValueError, match="illegal action"):
        env.step(np.int64(ACK))  # ACK is never legal during MAIN


def test_step_coerces_an_illegal_action_by_default() -> None:
    env = _make_env(on_illegal="coerce")
    env.reset(seed=1)

    _, _, terminated, _, _ = env.step(np.int64(ACK))

    assert not terminated  # coerced to a legal MAIN action, not applied raw


def test_terminal_step_returns_a_zero_observation() -> None:
    env = _make_env(step_limit=1)
    env.reset(seed=1)

    obs, _, terminated, _, _ = env.step(np.int64(PASS))

    assert terminated
    assert np.array_equal(obs, np.zeros(2, dtype=np.float32))


def test_opponent_pool_mode_builds_and_uses_one_agent_per_non_learner_seat() -> None:
    created: list[FakeAgent] = []

    def baseline_factory() -> FakeAgent:
        agent = FakeAgent("baseline")
        created.append(agent)
        return agent

    pool: OpponentPool = OpponentPool(
        ".",
        "*.zip",
        load_opponent=lambda path: FakeAgent("checkpoint"),
        baseline_factory=baseline_factory,
        rng=random.Random(0),
    )
    env = SingleAgentEnv(
        game=FakeGame(num_seats=3, step_limit=50),
        codec=FakeCodec(),
        encode=_encode,
        reward=_ScoreReward(),
        observation_space=_OBS_SPACE,
        num_seats=3,
        opponent_pool=pool,
        randomize_seat=False,
        seed=0,
    )

    env.reset(seed=1)
    # num_seats=3, learner at seat 0 -> exactly 2 non-learner seats.
    assert len(created) == 2

    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(np.int64(PASS))
        if terminated or truncated:
            break

    assert all(agent.choose_calls >= 1 for agent in created)
