"""The generic ``Agent`` protocol.

Method names and argument order are unchanged from the two implementations this
was extracted from -- ``catan/agents/base.py`` and ``truco-py/agents/base.py``,
which were independently written with the same call surface. The two type
parameters are the only addition: they are what lets a game's engine types stay
out of this package entirely (see the engine-vs-ML principle in catan's README
decision 10 -- ``gamekit`` imports from no game engine, ever).

``name`` is carried over from catan's variant, where it stamps results with the
role an agent played.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Agent[StateT, ActionT](Protocol):
    """Structural interface for all agents.

    ``name`` is a class-level annotation rather than a property on purpose:
    ``@runtime_checkable`` ``isinstance`` checks verify attribute *presence*,
    and a property would change what satisfying this protocol means.

    Note that ``isinstance`` only accepts the bare protocol -- ``isinstance(x,
    Agent)`` works, ``isinstance(x, Agent[S, A])`` raises ``TypeError``, as for
    any subscripted generic. Games that want a named, parameterised form should
    declare a type alias for annotations and keep ``isinstance`` on the bare
    import::

        type CatanAgent = Agent[GameState, Action]
    """

    name: str

    def choose_action(
        self, state: StateT, legal_actions: list[ActionT], player_idx: int
    ) -> ActionT:
        """Return one action from ``legal_actions`` for ``player_idx``."""
        ...

    def reset(self) -> None:
        """Called at the start of each new game. Override for stateful agents."""
        ...
