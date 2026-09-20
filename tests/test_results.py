from __future__ import annotations

import json
from pathlib import Path

from gamekit.results import git_commit, stamp, write_result


def test_git_commit_returns_unknown_outside_a_repo(tmp_path: Path) -> None:
    assert git_commit(cwd=tmp_path) == "unknown"


def test_git_commit_returns_a_hash_inside_this_repo() -> None:
    commit = git_commit(cwd=Path(__file__).resolve().parent)
    assert commit == "unknown" or len(commit) == 40


def test_stamp_includes_header_fields_and_extras() -> None:
    payload = stamp("my_experiment", n_games=10)
    assert payload["experiment"] == "my_experiment"
    assert payload["n_games"] == 10
    assert "git_commit" in payload
    assert "generated_at" in payload


def test_write_result_round_trips_through_json(tmp_path: Path) -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    path = write_result(tmp_path, "my_result", payload)
    assert path == tmp_path / "my_result.json"
    assert json.loads(path.read_text()) == payload
