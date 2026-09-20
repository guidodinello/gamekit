"""``SingleAgentEnv``: a gymnasium wrapper around a ``TurnBasedGame``, auto-
stepping every non-learner seat via ``gamekit.rl.driver.advance_until_learner``.

Extracted from ``truco-py/training/env.py``'s ``TrucoEnv``. This is the only
``gamekit.rl`` submodule that needs ``gymnasium``/``numpy`` -- the ``[rl]``
extra -- everything it delegates to (``protocols``, ``driver``, ``selfplay``,
``masking``) stays importable with no extra installed.

**Scope, stated honestly (gamekit#7 / catan Phase 5):** this env serves the
*flat, masked-``Discrete``* case -- one action index in, one mask vector out.
catan's action space is spatial and variable (a discard is a hand-sized
multiset choice; a trade proposal is an open-ended sentinel `legal_actions`
cannot enumerate), so catan's README concludes it needs a factored/
hierarchical action head, which this class does not provide. What catan (or
any variable-action-space game) reuses instead is ``protocols``, ``driver``
and ``selfplay`` directly, writing its own ``gym.Env`` around them -- that
split is the reason those three modules don't live inside this one.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from gamekit.agent import Agent
from gamekit.rl.driver import advance_until_learner
from gamekit.rl.masking import coerce_to_legal, legal_action_mask
from gamekit.rl.protocols import ActionCodec, RewardFn, TurnBasedGame
from gamekit.rl.selfplay import OpponentPool


def _place_fixed_agents[StateT, ActionT](
    agents: Sequence[Agent[StateT, ActionT]], num_seats: int, learner_seat: int
) -> list[Agent[StateT, ActionT] | None]:
    """Map a fixed, seat-order opponent list onto every non-learner seat --
    ``agents[0]`` goes to the lowest-numbered non-learner seat, and so on.
    Matches ``truco-py/training/env.py``'s ``_opponent_idx`` mapping exactly,
    so a fixed-opponent-list env behaves identically regardless of which
    seat the learner is assigned each episode.
    """
    result: list[Agent[StateT, ActionT] | None] = [None] * num_seats
    non_learner_seats = [s for s in range(num_seats) if s != learner_seat]
    for index, seat in enumerate(non_learner_seats):
        result[seat] = agents[index]
    return result


class SingleAgentEnv[StateT, ActionT](gym.Env[NDArray[np.float32], np.int64]):
    """Single-agent gymnasium env: the training agent is assigned a seat
    each episode (random by default), every other seat is auto-stepped by an
    ``Agent``, and each ``step()`` call corresponds to exactly one decision
    by the learner.

    Parameterized explicitly as ``gym.Env[NDArray[np.float32], np.int64]``
    (not left bare) -- gymnasium's ``Env`` is itself generic, and an
    unparameterized subclass degrades to ``Env[Any, Any]`` under mypy, the
    same failure mode ``docs/shared-ml-package.md`` records for the bare
    ``Agent`` protocol.

    Exactly one of ``agents`` (a fixed, seat-order opponent list) or
    ``opponent_pool`` (self-play resampling each episode) must be given.

    ``on_illegal`` matches truco-py's default exactly: ``"coerce"`` replaces
    an illegal action index with a uniformly random legal one, drawing from
    this env's own RNG -- a real policy choice, not a law, so ``"raise"`` is
    also available for callers who'd rather fail loudly.
    """

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        *,
        game: TurnBasedGame[StateT, ActionT],
        codec: ActionCodec[ActionT],
        encode: Callable[[StateT, int], NDArray[np.float32]],
        reward: RewardFn[StateT],
        observation_space: spaces.Box,
        num_seats: int,
        agents: Sequence[Agent[StateT, ActionT]] | None = None,
        opponent_pool: OpponentPool[StateT, ActionT] | None = None,
        randomize_seat: bool = True,
        on_illegal: Literal["coerce", "raise"] = "coerce",
        step_budget: int | None = None,
        seed: int | None = None,
    ) -> None:
        if (agents is None) == (opponent_pool is None):
            raise ValueError("pass exactly one of `agents` or `opponent_pool`")
        if agents is not None and len(agents) != num_seats - 1:
            raise ValueError(
                f"`agents` must have num_seats - 1 = {num_seats - 1} entries "
                f"(one per non-learner seat), got {len(agents)}"
            )
        if on_illegal not in ("coerce", "raise"):
            raise ValueError(
                f"on_illegal must be 'coerce' or 'raise', got {on_illegal!r}"
            )
        if observation_space.shape is None:
            raise ValueError("observation_space must have a concrete shape")

        self._game = game
        self._codec = codec
        self._encode = encode
        self._reward = reward
        self._num_seats = num_seats
        self._fixed_agents = agents
        self._opponent_pool = opponent_pool
        self._randomize_seat = randomize_seat
        self._on_illegal = on_illegal
        self._step_budget = step_budget
        self._rng = random.Random(seed)

        self.observation_space = observation_space
        self.action_space = spaces.Discrete(codec.n_actions)
        self._obs_shape: tuple[int, ...] = observation_space.shape

        self._state: StateT | None = None
        self._learner_seat: int = 0
        self._seat_agents: list[Agent[StateT, ActionT] | None] = []
        self._episode_steps = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset, preserving ``truco-py/training/env.py``'s exact draw order
        (gamekit#7, decision D2): reseed -> pick the learner seat -> resample
        or place opponents -> ``agent.reset()`` on each -> deal ->
        ``on_episode_start`` -> advance opponents, re-dealing if the game
        ends before the learner ever gets to act. Every draw comes from one
        ``random.Random`` stream, so this order is part of the contract, not
        an implementation detail -- reordering it silently changes training
        data for anything relying on a truco-py-derived seed.

        Keyword-only ``seed``/``options``, matching ``gymnasium.Env.reset``'s
        own (keyword-only) signature -- truco-py's positional ``reset`` was
        an LSP violation against the typed base.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        self._learner_seat = (
            self._rng.randint(0, self._num_seats - 1) if self._randomize_seat else 0
        )

        if self._opponent_pool is not None:
            seat_agents = self._opponent_pool.seat_agents(
                self._num_seats, self._learner_seat
            )
        else:
            assert self._fixed_agents is not None  # enforced in __init__
            seat_agents = _place_fixed_agents(
                self._fixed_agents, self._num_seats, self._learner_seat
            )
        for agent in seat_agents:
            if agent is not None:
                agent.reset()
        self._seat_agents = seat_agents

        while True:
            state = self._game.reset(seed=self._rng.randint(0, 2**31 - 1))
            self.on_episode_start(state, self._learner_seat)
            advance_until_learner(self._game, state, seat_agents)
            if not self._game.is_terminal(state):
                break

        self._state = state
        self._episode_steps = 0
        obs = self._encode(state, self._learner_seat)
        return obs, {}

    def step(
        self, action: np.int64
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one learner action; auto-advance every non-learner seat
        until the learner must act again or the episode ends."""
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        if self._game.is_terminal(self._state):
            raise RuntimeError("step() called on an already-terminal episode")

        mask = self._legal_mask()
        index = int(action)
        if not mask[index]:
            if self._on_illegal == "raise":
                raise ValueError(f"illegal action index {index}")
            index = coerce_to_legal(index, mask, self._rng)

        action_obj = self._codec.from_index(index)
        state = self._game.apply_action(self._state, action_obj)
        self._episode_steps += 1

        advance_until_learner(self._game, state, self._seat_agents)
        self._state = state

        terminated = self._game.is_terminal(state)
        truncated = (
            not terminated
            and self._step_budget is not None
            and self._episode_steps >= self._step_budget
        )
        reward = self._reward.compute(state, self._learner_seat, terminated)

        if terminated or truncated:
            obs = np.zeros(self._obs_shape, dtype=np.float32)
        else:
            obs = self._encode(state, self._learner_seat)

        return obs, reward, terminated, truncated, {"state": self._state}

    def action_masks(self) -> NDArray[np.bool_]:
        """Boolean mask of legal actions. Named for sb3-contrib's
        ``ActionMasker`` convention -- this module never imports sb3-contrib;
        the method exists to satisfy that convention structurally, exactly as
        truco-py's ``ActionMasker(env, lambda e: e.action_masks())`` expects."""
        return np.array(self._legal_mask(), dtype=np.bool_)

    def on_episode_start(self, state: StateT, seat: int) -> None:
        """Called once per episode, right after the deal, before any
        opponent acts. Default: forwards to the reward function's own hook.

        Override to add per-episode bookkeeping that a reward function or
        encoder needs but that isn't part of either protocol -- e.g.
        truco-py's ``_compute_v_mc`` (a Monte Carlo win-probability estimate
        from the initial deal, stashed for both the encoder closure and the
        reward shaper). Call ``super().on_episode_start(state, seat)`` too,
        unless the override also means to skip notifying the reward function.
        """
        self._reward.on_episode_start(state, seat)

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _legal_mask(self) -> list[bool]:
        if self._state is None or self._game.is_terminal(self._state):
            return [False] * self._codec.n_actions
        return legal_action_mask(self._game.legal_actions(self._state), self._codec)
