"""Self-play opponent pool: resample opponents from a directory of
checkpoints at each episode, with a floor probability of falling back to a
fixed baseline agent so training doesn't collapse into exploiting its own
past selves exclusively.

Extracted from ``truco-py/training/self_play.py`` and the resampling half of
``training/env.py:_resample_opponents``. Deliberately stdlib-only: loading a
checkpoint *file* into a model is framework-specific (``sb3_contrib`` in
truco-py's case) and stays with the game -- this module only decides *which
path* to hand to a caller-supplied loader, and how often to hand out the
baseline instead. See ``gamekit#7``'s dependency split (``gamekit.rl.env`` is
the only submodule that needs ``gymnasium``/``numpy``; this one needs neither).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

from gamekit.agent import Agent


class OpponentPool[StateT, ActionT]:
    """Scans ``directory`` for files matching ``pattern`` and samples one
    per call, mixing in ``baseline_factory()`` with probability
    ``baseline_mix``.

    ``load_opponent`` turns a checkpoint path into an ``Agent`` -- typically
    a lazy, cached wrapper (see truco-py's ``_CheckpointAgent``, which loads
    on first use and caches by path so a ``SubprocVecEnv`` worker loads each
    checkpoint at most once). This pool never imports a training framework;
    it only globs paths and asks the caller to turn one into an agent.

    Falls back to ``baseline_factory()`` unconditionally when the directory
    has no matching checkpoints yet (early training), matching truco-py's
    "falls back to ThresholdAgent when no checkpoints exist" behaviour.
    """

    def __init__(
        self,
        directory: str | Path,
        pattern: str,
        *,
        load_opponent: Callable[[Path], Agent[StateT, ActionT]],
        baseline_factory: Callable[[], Agent[StateT, ActionT]],
        baseline_mix: float = 0.2,
        rng: random.Random | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._pattern = pattern
        self._load_opponent = load_opponent
        self._baseline_factory = baseline_factory
        self._baseline_mix = baseline_mix
        self._rng = rng if rng is not None else random.Random()

    def _checkpoints(self) -> list[Path]:
        return sorted(self._directory.glob(self._pattern))

    def sample(self) -> Agent[StateT, ActionT]:
        """One opponent: a checkpoint agent, or the baseline per
        ``baseline_mix`` (or unconditionally, if no checkpoints exist yet)."""
        checkpoints = self._checkpoints()
        if checkpoints and self._rng.random() > self._baseline_mix:
            path = self._rng.choice(checkpoints)
            return self._load_opponent(path)
        return self._baseline_factory()

    def seat_agents(
        self, num_seats: int, learner_seat: int
    ) -> list[Agent[StateT, ActionT] | None]:
        """One freshly-sampled opponent per non-learner seat, ``None`` at
        ``learner_seat`` -- the exact shape ``gamekit.rl.driver.advance_until_learner``
        expects."""
        return [
            None if seat == learner_seat else self.sample() for seat in range(num_seats)
        ]
