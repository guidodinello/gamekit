"""The structural interfaces ``gamekit.rl`` drives a game through.

Stdlib-only, on purpose -- ``driver`` and ``selfplay`` need nothing else, and
keeping these Protocols here (rather than importing them from ``env``) is what
lets ``gamekit.rl.protocols``, ``.driver`` and ``.selfplay`` stay importable
with no extra installed.

``TurnBasedGame`` matches catan's ``CatanGame`` + the free function
``acting_player`` from ``engine/state.py``, and truco-py's ``TrucoGame`` +
``GameState.current_player``, as-is -- neither engine's public surface needs
to change, only a thin per-game adapter that satisfies this Protocol. See
``gamekit#7`` for why the seam is ``acting_player``, not "whose turn is it":
catan's robber-discard queue and trade-response round-trips mean the player
who must decide next is not always the turn player.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class TurnBasedGame[StateT, ActionT](Protocol):
    """A game engine driveable by ``gamekit.rl``.

    ``apply_action`` may mutate ``state`` in place and return that same
    object -- both catan's and truco-py's engines do this, and the driver
    never assumes it receives a copy. Callers that need an unmodified
    original must clone ``state`` themselves before calling.
    """

    def reset(self, seed: int | None = None) -> StateT:
        """Start a new game, returning the initial state."""
        ...

    def legal_actions(self, state: StateT) -> list[ActionT]:
        """Every action ``acting_player(state)`` may take right now."""
        ...

    def apply_action(self, state: StateT, action: ActionT) -> StateT:
        """Apply one action, returning the resulting state."""
        ...

    def is_terminal(self, state: StateT) -> bool:
        """Whether the game has ended."""
        ...

    def acting_player(self, state: StateT) -> int:
        """The seat that must act now -- not necessarily the turn player."""
        ...


@runtime_checkable
class ActionCodec[ActionT](Protocol):
    """Converts between a game's own action objects and a flat ``Discrete``
    index, for games whose legal-action set is small enough to enumerate.

    ``n_actions`` is a class- or instance-level attribute (not a property,
    for the same ``isinstance``-friendliness reason as ``gamekit.Agent.name``
    -- see that module's docstring).
    """

    n_actions: int

    def to_index(self, action: ActionT) -> int:
        """The flat index for ``action``."""
        ...

    def from_index(self, index: int) -> ActionT:
        """The action ``index`` decodes to."""
        ...


@runtime_checkable
class RewardFn[StateT](Protocol):
    """A reward function for one seat's episode.

    Deliberately a Protocol, not a base class to subclass -- ``gamekit.rl``
    does not ship a ``RewardShaper`` hierarchy; truco-py's ``SparseReward`` /
    ``ShapedReward`` / ``MCPotentialReward`` stay exactly where they are.
    """

    def compute(self, state: StateT, seat: int, done: bool) -> float:
        """The reward for ``seat`` observing ``state``; ``done`` marks the
        terminal step."""
        ...

    def on_episode_start(self, state: StateT, seat: int) -> None:
        """Called once per episode, after the deal and before any opponent
        acts. Override for reward functions that need an episode-start
        signal (e.g. an MC-estimated win probability). Default: no-op."""
