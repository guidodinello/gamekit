"""gamekit: game-agnostic agents, Monte Carlo statistics, and a benchmark
harness for game-simulation/ML projects.

Only ``Agent`` is re-exported at the top level. Everything else lives in its
own module (``gamekit.seats``, ``gamekit.results``, ``gamekit.benchmark``,
``gamekit.mc``) so importers name what they use.
"""

from __future__ import annotations

from gamekit.agent import Agent

__all__ = ["Agent"]
