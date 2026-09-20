"""Advance a game through every seat that isn't the learner.

Generalizes catan's ``server/bots.py:step_bots`` (loop on ``acting_player``,
stop when the acting seat has no agent) to any ``TurnBasedGame`` -- and fixes
the seat-index remapping ``truco-py/training/env.py:_opponent_idx`` needed
because that env kept opponents in a *5-element list excluding the learner's
seat*, rather than a *``num_seats``-element list keyed by seat directly*. This
module uses the direct keying: ``agents[seat] is None`` means "the learner sits
here," full stop, and there is nothing to remap.
"""

from __future__ import annotations

from collections.abc import Sequence

from gamekit.agent import Agent
from gamekit.rl.protocols import TurnBasedGame


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
