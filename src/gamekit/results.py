"""Result stamping: git commit, timestamp, and JSON-on-disk helpers.

Extracted from catan's ``experiments/exp_placement.py`` / ``experiments/exp_tables.py``,
which had the same three helpers duplicated verbatim, plus a private
cross-module import of them from ``experiments/benchmark.py``. Behaviour is
unchanged; ``RESULTS_DIR`` becomes a parameter instead of a module constant so
each game points this at its own ``experiments/results/`` directory.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def git_commit(cwd: Path | None = None) -> str:
    """The current commit hash, or ``"unknown"`` if it can't be determined
    (not a git repo, git not installed, or any other failure)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def stamp(experiment: str, **fields: Any) -> dict[str, Any]:
    """A result payload header: ``git_commit``, ``generated_at`` (UTC,
    ISO-8601), ``experiment``, plus whatever else the caller passes."""
    return {
        "git_commit": git_commit(),
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment": experiment,
        **fields,
    }


def write_result(results_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as indented JSON to ``results_dir / f"{name}.json"``,
    creating ``results_dir`` if needed. Returns the written path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
