# gamekit

Game-agnostic agents, Monte Carlo statistics, and a benchmark harness for
game-simulation/ML projects.

Extracted from [catan](https://github.com/guidodinello/catan)'s
`agents/base.py`, `experiments/mcstats.py`, and the game-agnostic parts of
`experiments/benchmark.py`, with `truco-py` as the second reference
implementation and mmo-utils' `DESIGN.md` folded in as requirements for the
`mc` module. See `~/projects/docs/shared-ml-package.md` for the full design
history.

## Modules

- `gamekit.agent` — the `Agent[StateT, ActionT]` protocol.
- `gamekit.mc` — confidence intervals, sample-size formulas, two-proportion
  testing, and streaming accumulators (stdlib-only).
- `gamekit.seats` — seat-keyed RNG streams and lineup rotation.
- `gamekit.results` — git-commit-stamped JSON result files.
- `gamekit.benchmark` — a field-free benchmark-arm runner: mandatory seat
  rotation, win-rate summaries by role and by seat, a two-proportion
  comparison, and a CLI skeleton. Never names a game's own record type.

## Scope

Core is stdlib + nothing else — no runtime dependencies. An `[rl]` extra
(torch/gymnasium/stable-baselines3) is planned but not yet built; see this
repo's issues for the rest of the v0.2 backlog (unified vectorized/iterative
Monte Carlo sampling, variance reduction, adaptive stopping, the `seat_rng`
explicit-mix migration).

## Installation

As a git dependency (the same pattern used by
[`guidodinello/logger`](https://github.com/guidodinello/logger)):

```toml
[project]
dependencies = ["gamekit"]

[tool.uv.sources]
gamekit = { git = "https://github.com/guidodinello/gamekit", branch = "main" }
```
