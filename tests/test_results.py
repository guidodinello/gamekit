from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

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


def test_write_result_failure_keeps_the_previous_file_and_leaves_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_result(tmp_path, "r", {"v": 1})

    def boom(src: object, dst: object) -> None:
        raise OSError("simulated crash")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        write_result(tmp_path, "r", {"v": 2})
    assert json.loads(path.read_text()) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["r.json"]


def test_concurrent_write_results_never_tear_the_file(tmp_path: Path) -> None:
    payloads = [{"writer": i, "pad": "x" * 50_000} for i in range(8)]
    threads = [
        threading.Thread(target=lambda p=p: write_result(tmp_path, "r", p))
        for p in payloads
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert json.loads((tmp_path / "r.json").read_text()) in payloads
    assert [p.name for p in tmp_path.iterdir()] == ["r.json"]
