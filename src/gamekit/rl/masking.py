"""Legal-action masking, stdlib-only.

Returns a plain ``list[bool]``, not an ndarray, on purpose: CI installs
gamekit's dev dependency group only (``pre-commit``, ``ruff``, ``mypy``,
``pytest``), which pulls in no numpy, so a numpy-importing test module here
would fail to *collect* rather than skip. ``gamekit.rl.env.action_masks()``
does the one-line ``np.fromiter(mask, dtype=bool)`` conversion at the
sb3-contrib boundary, where numpy is already a hard dependency.
"""

from __future__ import annotations

import random

from gamekit.rl.protocols import ActionCodec


def legal_action_mask[ActionT](
    legal_actions: list[ActionT], codec: ActionCodec[ActionT]
) -> list[bool]:
    """A length-``codec.n_actions`` boolean mask, ``True`` at the index of
    every action in ``legal_actions``."""
    mask = [False] * codec.n_actions
    for action in legal_actions:
        mask[codec.to_index(action)] = True
    return mask


def coerce_to_legal(index: int, mask: list[bool], rng: random.Random) -> int:
    """If ``index`` is legal under ``mask``, return it unchanged; otherwise
    return a uniformly random legal index.

    Matches ``truco-py/training/env.py:154-158``'s illegal-action policy
    exactly, including that a coercion consumes one draw from ``rng`` -- this
    is what ``gamekit.rl.env.SingleAgentEnv``'s ``on_illegal="coerce"`` (the
    default) calls.
    """
    if 0 <= index < len(mask) and mask[index]:
        return index
    legal_indices = [i for i, ok in enumerate(mask) if ok]
    return rng.choice(legal_indices)
