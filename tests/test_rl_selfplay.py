"""Tests for gamekit.rl.selfplay.OpponentPool -- stdlib only, runs
unconditionally in CI (see gamekit#7's D5)."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest
from _fakegame import FakeAgent

from gamekit.rl.selfplay import OpponentPool


def test_sample_falls_back_to_baseline_when_no_checkpoints_exist(
    tmp_path: Path,
) -> None:
    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent(f"checkpoint:{path.name}"),
        baseline_factory=lambda: FakeAgent("baseline"),
        baseline_mix=0.2,
        rng=random.Random(0),
    )

    assert pool.sample().name == "baseline"


def test_sample_picks_a_checkpoint_when_present_and_baseline_mix_is_zero(
    tmp_path: Path,
) -> None:
    (tmp_path / "ckpt_1.zip").touch()
    (tmp_path / "ckpt_2.zip").touch()
    (tmp_path / "other.txt").touch()  # must not match the glob pattern
    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent(f"checkpoint:{path.name}"),
        baseline_factory=lambda: FakeAgent("baseline"),
        baseline_mix=0.0,
        rng=random.Random(0),
    )

    assert pool.sample().name.startswith("checkpoint:ckpt_")


def test_sample_always_falls_back_to_baseline_when_baseline_mix_is_one(
    tmp_path: Path,
) -> None:
    (tmp_path / "ckpt_1.zip").touch()
    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent("checkpoint"),
        baseline_factory=lambda: FakeAgent("baseline"),
        baseline_mix=1.0,
        rng=random.Random(0),
    )

    # rng.random() draws from [0, 1) and is never > baseline_mix=1.0, so
    # every draw takes the baseline branch even with checkpoints present.
    for _ in range(20):
        assert pool.sample().name == "baseline"


def test_seat_agents_places_none_at_the_learner_seat_only(tmp_path: Path) -> None:
    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent("checkpoint"),
        baseline_factory=lambda: FakeAgent("baseline"),
        rng=random.Random(0),
    )

    agents = pool.seat_agents(num_seats=4, learner_seat=2)

    assert len(agents) == 4
    assert agents[2] is None
    assert all(agent is not None for seat, agent in enumerate(agents) if seat != 2)


def test_checkpoint_dir_defaults_to_the_given_directory(tmp_path: Path) -> None:
    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent(path.name),
        baseline_factory=lambda: FakeAgent("baseline"),
    )

    assert pool.checkpoint_dir == tmp_path


def test_checkpoint_dir_is_the_run_id_subdirectory(tmp_path: Path) -> None:
    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent(path.name),
        baseline_factory=lambda: FakeAgent("baseline"),
        run_id="run-a",
    )

    assert pool.checkpoint_dir == tmp_path / "run-a"


def test_run_scoped_pool_ignores_checkpoints_outside_its_run(tmp_path: Path) -> None:
    (tmp_path / "ckpt_0.zip").touch()
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "ckpt_1.zip").touch()

    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent(path.name),
        baseline_factory=lambda: FakeAgent("baseline"),
        baseline_mix=0.0,
        run_id="run-a",
        rng=random.Random(0),
    )

    assert pool.checkpoints() == [run_dir / "ckpt_1.zip"]
    for _ in range(20):
        assert pool.sample().name == "ckpt_1.zip"


def test_stale_final_checkpoint_from_a_collapsed_run_is_never_sampled(
    tmp_path: Path,
) -> None:
    """Regression for gamekit#23: a stale `*_final.zip` left over from an
    earlier, collapsed run must not be eligible once the pool is scoped to
    the current run -- and, to prove this is a real regression test and not
    a tautology about an empty directory, an *unscoped* pool over the exact
    same tree does still see the stale file."""
    (tmp_path / "truco_selfplay_final.zip").touch()  # the collapsed April run
    run_dir = tmp_path / "run-b"
    run_dir.mkdir()
    (run_dir / "truco_selfplay_1000.zip").touch()

    scoped: OpponentPool = OpponentPool(
        tmp_path,
        "truco_selfplay_*.zip",
        load_opponent=lambda path: FakeAgent(path.name),
        baseline_factory=lambda: FakeAgent("baseline"),
        baseline_mix=0.0,
        run_id="run-b",
        rng=random.Random(0),
    )
    for _ in range(50):
        assert "final" not in scoped.sample().name

    unscoped: OpponentPool = OpponentPool(
        tmp_path,
        "truco_selfplay_*.zip",
        load_opponent=lambda path: FakeAgent(path.name),
        baseline_factory=lambda: FakeAgent("baseline"),
    )
    assert tmp_path / "truco_selfplay_final.zip" in unscoped.checkpoints()


def test_checkpoints_lists_matching_paths_sorted_and_excludes_non_matches(
    tmp_path: Path,
) -> None:
    (tmp_path / "ckpt_2.zip").touch()
    (tmp_path / "ckpt_1.zip").touch()
    (tmp_path / "other.txt").touch()

    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent(path.name),
        baseline_factory=lambda: FakeAgent("baseline"),
    )

    assert pool.checkpoints() == [tmp_path / "ckpt_1.zip", tmp_path / "ckpt_2.zip"]


def test_checkpoints_is_empty_when_the_run_directory_does_not_exist(
    tmp_path: Path,
) -> None:
    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent(path.name),
        baseline_factory=lambda: FakeAgent("baseline"),
        run_id="never-created",
        rng=random.Random(0),
    )

    assert pool.checkpoints() == []
    assert pool.sample().name == "baseline"


def test_pool_never_creates_the_run_directory(tmp_path: Path) -> None:
    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent(path.name),
        baseline_factory=lambda: FakeAgent("baseline"),
        run_id="run-a",
        rng=random.Random(0),
    )

    pool.sample()
    pool.checkpoints()

    assert not (tmp_path / "run-a").exists()


def test_checkpoints_reflects_files_written_after_construction(
    tmp_path: Path,
) -> None:
    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent(path.name),
        baseline_factory=lambda: FakeAgent("baseline"),
    )

    assert pool.checkpoints() == []

    (tmp_path / "ckpt_1.zip").touch()

    assert len(pool.checkpoints()) == 1


@pytest.mark.parametrize(
    "run_id",
    ["", "  ", ".", "..", "a/b", "nested/run", "/abs", "a\\b"],
)
def test_run_id_must_name_a_single_directory(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError):
        OpponentPool(
            tmp_path,
            "ckpt_*.zip",
            load_opponent=lambda path: FakeAgent(path.name),
            baseline_factory=lambda: FakeAgent("baseline"),
            run_id=run_id,
        )


def test_run_id_error_names_the_offending_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=re.escape(repr("a/b"))):
        OpponentPool(
            tmp_path,
            "ckpt_*.zip",
            load_opponent=lambda path: FakeAgent(path.name),
            baseline_factory=lambda: FakeAgent("baseline"),
            run_id="a/b",
        )


def test_seat_agents_uses_the_run_scoped_directory(tmp_path: Path) -> None:
    (tmp_path / "ckpt_0.zip").touch()  # outside the run -- must not be used
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "ckpt_1.zip").touch()

    pool: OpponentPool = OpponentPool(
        tmp_path,
        "ckpt_*.zip",
        load_opponent=lambda path: FakeAgent(path.name),
        baseline_factory=lambda: FakeAgent("baseline"),
        baseline_mix=0.0,
        run_id="run-a",
        rng=random.Random(0),
    )

    agents = pool.seat_agents(num_seats=4, learner_seat=2)

    assert agents[2] is None
    for seat, agent in enumerate(agents):
        if seat != 2:
            assert agent is not None
            assert agent.name == "ckpt_1.zip"
