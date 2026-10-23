"""Advance a game through every seat that isn't the learner.

Generalizes catan's ``server/bots.py:step_bots`` (loop on ``acting_player``,
stop when the acting seat has no agent) to any ``TurnBasedGame`` -- and fixes
the seat-index remapping ``truco-py/training/env.py:_opponent_idx`` needed
because that env kept opponents in a *5-element list excluding the learner's
seat*, rather than a *``num_seats``-element list keyed by seat directly*. This
module uses the direct keying: ``agents[seat] is None`` means "the learner sits
here," full stop, and there is nothing to remap.

Also home to the two helpers every single-agent env's ``reset`` shares --
``place_fixed_agents`` (seat mapping) and ``deal_until_learner`` (the bounded
re-deal loop) -- so the draw order and the loop bound live in one place
instead of being copied per env (gamekit#55).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from gamekit.agent import Agent
from gamekit.rl.protocols import TurnBasedGame

MAX_REDEALS_DEFAULT = 100


def advance_until_learner[StateT, ActionT](
    game: TurnBasedGame[StateT, ActionT],
    state: StateT,
    agents: Sequence[Agent[StateT, ActionT] | None],
) -> int:
    """Step every seat whose ``agents[seat]`` is not ``None``, in order,
    until the acting seat's agent is ``None`` (the learner's turn) or the
    game reaches a terminal state.

    ``agents`` is indexed by seat directly -- ``agents[game.acting_player(state)]``
    is always the right lookup, with no separate opponent-index mapping.
    Mutates ``state`` in place (via ``game.apply_action``) and returns the
    number of steps taken.
    """
    steps = 0
    while not game.is_terminal(state):
        seat = game.acting_player(state)
        agent = agents[seat]
        if agent is None:
            break
        legal = game.legal_actions(state)
        if not legal:
            break
        action = agent.choose_action(state, legal, seat)
        state = game.apply_action(state, action)
        steps += 1
    return steps


def place_fixed_agents[StateT, ActionT](
    agents: Sequence[Agent[StateT, ActionT]], num_seats: int, learner_seat: int
) -> list[Agent[StateT, ActionT] | None]:
    """Map a fixed, seat-order opponent list onto every non-learner seat --
    ``agents[0]`` goes to the lowest-numbered non-learner seat, and so on;
    ``None`` marks the learner's seat. Matches ``truco-py/training/env.py``'s
    ``_opponent_idx`` mapping, so a fixed-opponent-list env behaves the same
    whichever seat the learner is assigned each episode.

    Raises ``ValueError`` unless ``len(agents) == num_seats - 1`` and
    ``0 <= learner_seat < num_seats``.
    """
    if not 0 <= learner_seat < num_seats:
        raise ValueError(f"learner_seat {learner_seat} outside [0, {num_seats})")
    if len(agents) != num_seats - 1:
        raise ValueError(
            f"`agents` must have num_seats - 1 = {num_seats - 1} entries "
            f"(one per non-learner seat), got {len(agents)}"
        )
    result: list[Agent[StateT, ActionT] | None] = [None] * num_seats
    non_learner_seats = [s for s in range(num_seats) if s != learner_seat]
    for index, seat in enumerate(non_learner_seats):
        result[seat] = agents[index]
    return result


def deal_until_learner[StateT, ActionT](
    game: TurnBasedGame[StateT, ActionT],
    rng: random.Random,
    seat_agents: Sequence[Agent[StateT, ActionT] | None],
    *,
    on_deal: Callable[[StateT], None] | None = None,
    max_redeals: int = MAX_REDEALS_DEFAULT,
) -> StateT:
    """Deal, let the opponents act, and re-deal if the game ends before the
    learner ever gets to act. Returns the first non-terminal state.

    Each attempt is, in this order: ``game.reset(seed=rng.randint(0, 2**31 - 1))``
    -> ``on_deal(state)`` -> ``advance_until_learner``. The order is part of
    the reset contract (gamekit#7, D2): it fixes how many draws come off
    ``rng``. ``on_deal`` fires once per attempt, including discarded ones.

    ``max_redeals`` counts total deals, so the default 100 means at most 100
    attempts. If every one ends before the learner acts, this raises
    ``RuntimeError`` rather than spinning forever -- the usual cause is a
    misconfigured opponent set. ``max_redeals < 1`` raises ``ValueError``.
    """
    if max_redeals < 1:
        raise ValueError(f"max_redeals must be >= 1, got {max_redeals}")
    for _ in range(max_redeals):
        state = game.reset(seed=rng.randint(0, 2**31 - 1))
        if on_deal is not None:
            on_deal(state)
        advance_until_learner(game, state, seat_agents)
        if not game.is_terminal(state):
            return state
    raise RuntimeError(
        f"every one of {max_redeals} deals ended before the learner acted "
        f"-- check the opponent configuration"
    )
