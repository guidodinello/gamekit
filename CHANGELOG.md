# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking
- `gamekit.seats.seat_rng` now derives its seed from a BLAKE2b digest of
  `(driver_seed, seat)` instead of `hash((driver_seed, seat))`. This changes
  every stream `seat_rng` produces, so any committed result generated under
  the old stream is no longer reproducible from `seat_rng` alone -- either
  regenerate it, or replay it via `seat_rng_legacy`. See issue #8.

### Added
- `gamekit.seats.seat_rng_legacy` -- the pre-0.2.0 `hash()`-based stream,
  kept so already-committed results stay replayable.
- Initial project bootstrap with dev-standards baseline
- `gamekit.rl`: an `[rl]` extra (gymnasium + numpy) providing the reusable
  structure of a single-agent RL training loop — `SingleAgentEnv`
  (`gym.Env` subclass, auto-stepping non-learner seats), legal-action
  masking, and self-play opponent resampling with a baseline-mix fallback.
  Extracted from `truco-py`'s `training/env.py`; see `gamekit.rl.env`'s
  module docstring and `~/projects/docs/shared-ml-package.md` for the
  design history. Closes [#7](https://github.com/guidodinello/gamekit/issues/7).
- `gamekit.mc.sample`: a unified `Sampler -> Evaluator -> Accumulator`
  Monte Carlo fold (`monte_carlo_reduce`), with `monte_carlo` as a
  convenience shortcut over it (mmo-utils DESIGN.md gap 1, #4). Stdlib-only
  -- no numpy dependency, no `[numpy]` extra; a vectorized sampler returning
  a numpy `ndarray` satisfies `Sampler[T]` structurally.
- `Typecheck Python` CI job (mypy, 3.13/3.14 matrix)

### Changed
- `gamekit.benchmark`'s docstrings clarify that a "seat" is a competitor slot
  (a single player or a fixed team), not necessarily one player (#11)

### Deprecated
- `gamekit.seats.seat_rng_legacy` is deprecated on arrival: it exists only
  to replay pre-0.2.0 results and should not be used in new code.

## [0.1.0] - 2026-09-19

### Added
- Project scaffolded