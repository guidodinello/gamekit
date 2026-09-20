"""A minimal 3-seat toy game for ``gamekit.rl`` tests. Stdlib only.

Deliberately includes one **out-of-turn decision phase** -- an ACK queue
triggered by a CLAIM -- modeled on catan's post-seven discard queue
(``engine/game.py:506-521``), so that ``gamekit.rl``'s ``acting_player``-driven
advance loop (gamekit#7, decision D3a) is actually exercised by a test, not
just asserted in a docstring.

Turn structure:
    MAIN phase -- ``acting_player`` is ``turn_player``. Legal actions are
    ``CLAIM`` (score a point, then every *other* seat owes an ACK) or
    ``PASS`` (advance the turn immediately).

    ACK phase -- once a CLAIM queues it, ``acting_player`` is the next seat
    still owing an acknowledgement (``pending[0]``), regardless of turn
    order. The only legal action is ``ACK``. Once every seat has
    acknowledged, the turn advances.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

PASS = 0
CLAIM = 1
ACK = 2


@dataclasses.dataclass(slots=True)
class FakeState:
    num_seats: int
    turn_player: int = 0
    pending: list[int] = dataclasses.field(default_factory=list)
    scores: list[int] = dataclasses.field(default_factory=list)
    steps: int = 0
    step_limit: int = 12

    def __post_init__(self) -> None:
        if not self.scores:
            self.scores = [0] * self.num_seats


class FakeGame:
    """Satisfies ``gamekit.rl.protocols.TurnBasedGame[FakeState, int]``."""

    def __init__(self, num_seats: int = 3, step_limit: int = 12) -> None:
        self.num_seats = num_seats
        self.step_limit = step_limit

    def reset(self, seed: int | None = None) -> FakeState:
        del seed  # deterministic toy game -- nothing to seed
        return FakeState(num_seats=self.num_seats, step_limit=self.step_limit)

    def legal_actions(self, state: FakeState) -> list[int]:
        if self.is_terminal(state):
            return []
        if state.pending:
            return [ACK]
        return [PASS, CLAIM]

    def apply_action(self, state: FakeState, action: int) -> FakeState:
        if state.pending:
            if action != ACK:
                raise ValueError(f"illegal action {action} during ACK phase")
            state.pending.pop(0)
            if not state.pending:
                state.turn_player = (state.turn_player + 1) % self.num_seats
        elif action == CLAIM:
            state.scores[state.turn_player] += 1
            state.pending = [s for s in range(self.num_seats) if s != state.turn_player]
        elif action == PASS:
            state.turn_player = (state.turn_player + 1) % self.num_seats
        else:
            raise ValueError(f"illegal action {action} during MAIN phase")
        state.steps += 1
        return state

    def is_terminal(self, state: FakeState) -> bool:
        return state.steps >= state.step_limit

    def acting_player(self, state: FakeState) -> int:
        if state.pending:
            return state.pending[0]
        return state.turn_player


class FakeAgent:
    """A scripted ``Agent[FakeState, int]``: ``policy`` picks from
    ``legal_actions``, defaulting to "always take the first legal action."
    Records call counts for assertions."""

    def __init__(
        self,
        name: str,
        policy: Callable[[FakeState, list[int], int], int] | None = None,
    ) -> None:
        self.name = name
        self._policy = policy or (lambda state, legal, seat: legal[0])
        self.choose_calls = 0
        self.reset_calls = 0

    def choose_action(
        self, state: FakeState, legal_actions: list[int], player_idx: int
    ) -> int:
        self.choose_calls += 1
        return self._policy(state, legal_actions, player_idx)

    def reset(self) -> None:
        self.reset_calls += 1


def always_claim(state: FakeState, legal: list[int], seat: int) -> int:
    """A policy that claims whenever legal, otherwise acks -- reliably
    triggers the ACK out-of-turn phase for every other seat."""
    del state, seat
    return CLAIM if CLAIM in legal else legal[0]


class FakeCodec:
    """``ActionCodec[int]`` where the action *is* its own index -- MAIN's
    two actions (PASS/CLAIM) and ACK share one flat space of size 3."""

    n_actions = 3

    def to_index(self, action: int) -> int:
        return action

    def from_index(self, index: int) -> int:
        return index
