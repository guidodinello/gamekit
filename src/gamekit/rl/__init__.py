"""``gamekit.rl``: the reusable structure of a single-agent RL training loop --
turn-advancing, legal-action masking, self-play opponent resampling -- lifted
out of ``truco-py``'s ``training/env.py``. See ``~/projects/docs/shared-ml-package.md``
and `gamekit#7 <https://github.com/guidodinello/gamekit/issues/7>`_ for the
design history.

Nothing is re-exported here, deliberately, same as ``gamekit``'s own top level:
``gamekit.rl.protocols``, ``gamekit.rl.driver`` and ``gamekit.rl.selfplay`` are
stdlib-only and importable with no extra installed; ``gamekit.rl.env`` needs
``gymnasium`` and ``numpy`` (the ``[rl]`` extra). A re-export here would make
every submodule import fail the moment ``gymnasium`` is missing, which is
exactly the environment CI runs in.
"""

from __future__ import annotations
