"""Tests for gamekit.rl.selfplay.OpponentPool -- stdlib only, runs
unconditionally in CI (see gamekit#7's D5)."""

from __future__ import annotations

import random
from pathlib import Path

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
