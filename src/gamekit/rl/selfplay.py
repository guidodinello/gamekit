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

A pool is scoped to exactly one directory, and ``run_id`` names it: with
``run_id`` set, the pool globs ``directory / run_id``, and ``checkpoint_dir``
exposes that path so the checkpoint *writer* and this *reader* share one
source of truth (``model.save(pool.checkpoint_dir / f"ckpt_{step}.zip")``).
Without it (``run_id=None``, the default and the pre-0.3.0 behaviour) the
pool globs ``directory`` itself and samples every matching file there --
including checkpoints left by an earlier or collapsed run, which is how a
stale ``truco_selfplay_final.zip`` stayed an eligible opponent for every
later run (`gamekit#23 <https://github.com/guidodinello/gamekit/issues/23>`_).
Callers who cannot move files into a subdirectory can scope ``pattern``
instead (e.g. ``f"{run_id}_ckpt_*.zip"``). The pool never creates or deletes
anything under ``checkpoint_dir``: an absent or empty directory means "no
checkpoints yet" and falls back to the baseline, and ``checkpoints()`` lists
what is eligible right now, so a caller can log or assert on it at startup.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from pathlib import Path

from gamekit.agent import Agent

# "." / ".." would resolve *outside* `directory` -- the exact escape #23 is
# about -- and neither is caught by a naive "single path component" check via
# Path.parts (Path("..").parts == ("..",)).
_RESERVED_RUN_IDS = (os.curdir, os.pardir)
_RUN_ID_SEPARATORS = ("/", "\\")


def _checkpoint_directory(directory: Path, run_id: str | None) -> Path:
    """``directory``, or ``directory / run_id`` when ``run_id`` names a
    single, real subdirectory component."""
    if run_id is None:
        return directory
    if (
        not run_id.strip()
        or run_id in _RESERVED_RUN_IDS
        or any(sep in run_id for sep in _RUN_ID_SEPARATORS)
    ):
        raise ValueError(
            f"run_id must name a single directory under {directory}, got {run_id!r}"
        )
    return directory / run_id


class OpponentPool[StateT, ActionT]:
    """Scans ``checkpoint_dir`` for files matching ``pattern`` and samples
    one per call, mixing in ``baseline_factory()`` with probability
    ``baseline_mix``. See the module docstring for how ``run_id`` scopes
    ``checkpoint_dir`` to a single run.

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
        run_id: str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._checkpoint_dir = _checkpoint_directory(Path(directory), run_id)
        self._pattern = pattern
        self._load_opponent = load_opponent
        self._baseline_factory = baseline_factory
        self._baseline_mix = baseline_mix
        self._rng = rng if rng is not None else random.Random()

    @property
    def checkpoint_dir(self) -> Path:
        """The one directory this pool globs -- ``directory / run_id``, or
        ``directory`` when ``run_id`` is ``None``. Never created by the pool:
        an absent directory just means no checkpoints exist yet."""
        return self._checkpoint_dir

    def checkpoints(self) -> list[Path]:
        """Every path currently eligible for ``sample()``: ``checkpoint_dir``'s
        matches for ``pattern``, sorted. Re-globs on each call -- the set
        grows as training writes new checkpoints."""
        return sorted(self._checkpoint_dir.glob(self._pattern))

    def sample(self) -> Agent[StateT, ActionT]:
        """One opponent: a checkpoint agent, or the baseline per
        ``baseline_mix`` (or unconditionally, if no checkpoints exist yet)."""
        checkpoints = self.checkpoints()
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
