# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking
- `SingleAgentEnv`'s `step_budget` now counts engine actions (the learner's
  and every opponent's), not learner decisions, matching catan's `CatanEnv`.
  Anyone passing `step_budget` sees the same budget truncate about
  `num_seats` times sooner. The budget is still checked after the opponents
  advance, so an episode can overshoot it. truco-py never passes
  `step_budget` and is unaffected (#55).

### Added
- `gamekit.rl.driver.place_fixed_agents` and `deal_until_learner`: the seat
  mapping and the bounded re-deal loop of `SingleAgentEnv.reset` as public,
  stdlib-only helpers, so another env can share the reset contract instead of
  copying it. `SingleAgentEnv` now calls them; the RNG draw order is unchanged
  (#55).
- `SingleAgentEnv(max_redeals=...)` (default `MAX_REDEALS_DEFAULT = 100`).
- README: a "Monte Carlo statistics" section saying what each part of `gamekit.mc`
  is for (intervals and tests, sample size, the fold, adaptive stopping, variance
  reduction), with an example each. The `gamekit.mc` module docstring has the same
  map. The whole `mc` surface stays public (#53).
- `gamekit.league.run_pairing(a, b, ...)`: the loop body of `run_league` (config
  hash check, `run_arm`, atomic write) as a public function. It writes only the
  pairing file, never `league.json`, and needs no anchor, so a consumer can play
  pairings from its own process pool and call `summarize_league` once. It is
  order-insensitive in `(a, b)` and writes the same file, seeds and hash as
  `run_league`, so existing result directories keep resuming. `run_league` now
  calls it (#47).
- `.github/workflows/release.yml`: pushing a `vX.Y.Z` tag now creates the GitHub
  Release. It checks the tag against `pyproject.toml`'s version and `main`, uses
  that version's section of this file as the notes, and attaches the wheel and
  sdist built by `uv build`. See README `Releasing`.

### Fixed
- `SingleAgentEnv.reset` re-dealt in an unbounded loop; it now raises
  `RuntimeError` after `max_redeals` deals that all end before the learner
  acts (#55).
- `SingleAgentEnv.step` now treats an action index outside
  `[0, codec.n_actions)` as illegal and follows `on_illegal`. A negative
  index used to wrap around the mask and an index past the end raised
  `IndexError`. Under `"coerce"` such an index now consumes one RNG draw (#55).

### Changed
- The `SingleAgentEnv` docstring and README no longer say catan "needs a
  factored/hierarchical head" (its atom composer is one); the real reason it
  has its own env is that an env step is not always one engine action (#55).
- The `mc` helpers that sibling modules shared (`_NORMAL`, `_z_score`, `_drawn`) now
  live in a private `gamekit/mc/_common.py` instead of being imported across modules
  as underscore names. No public change: `gamekit.mc.__all__` is the same (#51).
- `tach.toml` now declares an `[[interfaces]]` entry for `gamekit.mc` exposing all
  of `gamekit.mc.__all__`, so other gamekit modules must import mc names from the
  package, not from a submodule. A test keeps the list in sync with `__all__` (#53).
- `gamekit.results.write_result` is now atomic: it writes a uniquely named temp
  file in the same directory and `os.replace`s it into place, so concurrent
  writers (e.g. parallel `run_league` calls writing `league.json`) cannot leave a
  torn file. This applies to every caller, not only the league. The bytes written
  are the same; `os.replace` replaces a symlink instead of writing through it,
  and a new file gets the umask-default mode (#47).
- `.pre-commit-config.yaml`: the mypy hook is now a local hook running
  `uv run --locked --extra rl mypy src tests`, so it type-checks against the
  versions in `uv.lock` (it used `mirrors-mypy` with floating `additional_dependencies`).
  CI's `pre-commit` job no longer sets `SKIP=mypy` (#63).
- `.github/workflows/release.yml`: its actions are now pinned to commit SHAs and
  checkout no longer persists the token, as the release job holds `contents: write`.

## [0.4.0] - 2026-10-10

### Added
- `gamekit.league`: a round-robin league with anchored ratings. `run_league`
  plays every pairing of named agents through `gamekit.benchmark.run_arm`
  (seat-rotated, `n` per pairing, seeds derived from `(seed, a, b)` so adding an
  agent never reshuffles existing pairings) and writes one resumable result file
  per pairing via `gamekit.results`; a re-run skips pairings whose stored config
  hash matches and raises on a changed config. `summarize_league` fits a
  Bradley-Terry MLE (stdlib-only Hunter MM iteration, no numpy; a weak prior keeps
  perfect scores finite) on an Elo scale pinned to a named anchor, with stratified
  bootstrap CIs, the pairwise win matrix with Wilson CIs and BT residuals, and a
  report of 3-cycles among significant edges. Multi-seat games use a balanced
  `(a, b, a, b, ...)` pairwise reduction; odd seat counts raise
  `NotImplementedError`. Additive only; nothing existing changes. Research note
  [022](docs/research/022-league-ratings.md) records the design and its evidence
  (#38).
- `docs/research/`: a git-tracked logbook of game-agnostic technique notes
  (hypothesis, citations, how to test, result), with a note template and an
  experiment-log template for consumer repos (#26, #27, #28). It now holds
  notes 001-023.
- README `Architecture` section and `docs/architecture.html`, a detailed page
  with the module index, what a game implements, and the data flows (#50).
- `tach.toml` and a blocking `Module Boundaries (Python)` CI job (plus a
  pre-commit hook) enforcing the import graph in the README's Architecture
  section; ruff's `TID251` now bans importing `numpy`/`gymnasium` outside
  `gamekit.rl.env` (#56).

### Changed
- **Typing change for consumers:** `SingleAgentEnv`'s `observation_space`
  parameter is now typed `spaces.Box[np.float32]` (it was an unparameterized
  `spaces.Box`), following gymnasium 1.4. There is no runtime change, but a
  type-checked consumer that passes an unparameterized or non-float32 `Box`
  may now get a mypy error; build the space with `dtype=np.float32` or
  annotate it as `Box[np.float32]` (#65).
- The `rl` extra now requires `gymnasium>=1.4.0` (was `>=1.3.0`) (#65).
- Dev tooling: ruff 0.16.10, mypy 2.4.0, and `uv.lock` resynced with
  `pyproject.toml` (#43, #44, #45, #64, #65).
- CI jobs pick their runner from the `CI_RUNNERS` repository variable,
  falling back to `ubuntu-latest` when unset (#25).
- CI hardening (#57): a `Packaging (wheel)` job installs the built wheel into
  clean venvs and asserts the core imports without numpy/gymnasium; every job
  syncs with `uv sync --locked`; `pre-commit` and `actionlint` run as jobs;
  superseded PR runs are cancelled; and an aggregate `CI` job gives the branch
  ruleset one stable check to require; the `main` ruleset now requires only
  that `CI` check. Dependabot now uses the `uv` ecosystem
  so it keeps `uv.lock` in step with `pyproject.toml`.

## [0.3.0] - 2026-09-20

### Added
- `gamekit.rl.selfplay.OpponentPool` gains a keyword-only `run_id`
  constructor argument that scopes the checkpoint glob to
  `directory / run_id`, a `checkpoint_dir` property exposing that resolved
  path so a checkpoint writer and this reader share one source of truth,
  and a public `checkpoints()` method listing every path currently eligible
  for `sample()`. `run_id` defaults to `None`, which keeps the pre-0.3.0
  behaviour of globbing `directory` itself unchanged. See #23.

### Fixed
- `OpponentPool` globbed the whole checkpoint directory with no notion of
  which run produced a file, so a checkpoint left over from an earlier or
  collapsed run stayed eligible for sampling forever. truco-py hit this in
  practice: a `truco_selfplay_final.zip` from a run that had collapsed to
  ~57% vs `ThresholdAgent` (down from 85.3%) kept getting sampled as an
  opponent by every later run. Passing `run_id` scopes the pool to a single
  run's subdirectory, so a stale checkpoint from a different run can no
  longer be drawn. Closes #23.

### Changed
- CI gains a dedicated, required `RL extra` job that installs the `[rl]`
  extra and runs pytest and mypy against real gymnasium/numpy (#22).

## [0.2.0] - 2026-09-20

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
- `gamekit.mc.variance`: antithetic variates (`antithetic` + `paired_mean`)
  and control variates (`control_variate` + `control_beta`) as
  Sampler/Evaluator wrappers over the gap-1 fold (mmo-utils DESIGN.md gap 4,
  #5). Each is tested against a known integrand with negative controls that
  isolate the correlation effect from the arithmetic of averaging.
- `gamekit.mc.stopping.monte_carlo_until`: run the gap-1 fold in chunks
  until a target confidence-interval half-width is reached, with a
  mandatory `max_n` hard cap (mmo-utils DESIGN.md gap 5, #6). Not wired
  into `gamekit.benchmark.run_arm` -- its exact-multiple-of-lineup-length
  requirement is a real constraint on any future integration, noted but
  not designed here.

### Changed
- `gamekit.benchmark`'s docstrings clarify that a "seat" is a competitor slot
  (a single player or a fixed team), not necessarily one player (#11)

### Deprecated
- `gamekit.seats.seat_rng_legacy` is deprecated on arrival: it exists only
  to replay pre-0.2.0 results and should not be used in new code.

## [0.1.0] - 2026-09-19

### Added
- Project scaffolded

[Unreleased]: https://github.com/guidodinello/gamekit/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/guidodinello/gamekit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/guidodinello/gamekit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/guidodinello/gamekit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/guidodinello/gamekit/releases/tag/v0.1.0
