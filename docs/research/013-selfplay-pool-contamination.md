# 013 — Self-play pool contamination across runs

**Status:** validated
**Last touched:** 2026-09-21

## Hypothesis

A checkpoint pool that resolves eligibility by globbing a shared directory
lets stale checkpoints from an earlier, already-collapsed run stay
eligible for `sample()` forever, contaminating the opponent distribution
of every later run in proportion to how many stale files are lying
around — and this does not wash out with more training. This is distinct
from [001](001-self-play-opponent-mix.md)'s question of *how much* fixed
baseline share to mix in: here the broken variable is *which checkpoints
are even in the pool*, not the ratio applied to them.

## Why we believe it

- gamekit issue #23 names the mechanism directly: "There is no notion of
  *which run* produced a checkpoint. Any file in `directory` matching
  `pattern` is eligible for `sample()` forever, including checkpoints
  written by an earlier, unrelated, or collapsed run" — no external
  source; observed in gamekit issue #23.
- truco-py hit this in practice, twice, at two different mix values. The
  April run (default `threshold_mix=0.2`) collapsed from 85.3% to 56-57.3%
  vs `ThresholdAgent`; the June run (`--threshold-mix 0.5`, a heavier
  baseline share chosen specifically in response to the April collapse)
  collapsed again, `ep_rew_mean` going from ~0.4 to -7.28 and `ep_len_mean`
  from ~2 to 165 within about 11 minutes of wall clock — no external
  source; observed in truco-py logs
  [004](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/004-april-selfplay-collapse.md)
  and
  [005](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/005-june-threshold-mix-collapse.md).
  Raising the mix from 0.2 to 0.5 did not prevent the second collapse,
  which is exactly what a pool-contamination root cause predicts and a
  mix-ratio root cause does not.
- The confirmed root cause, gamekit PR #24: `OpponentPool` sampled
  uniformly over every file matching `truco_selfplay_*.zip` in a shared
  `checkpoints/` directory, so a stale checkpoint from the April run
  (`truco_selfplay_final.zip`, already collapsed to ~57%) stayed in the
  sample pool for the June run too — no external source; observed in
  gamekit PR #24's description, which names truco-py explicitly.
- catan's own self-play attempts used a fresh, run-scoped pool from the
  start (`run_id=cfg.label`, gamekit 0.3.0) and never exhibited this
  failure mode at either mix value tested — no external source; observed
  in catan log
  [003](https://github.com/guidodinello/catan/blob/main/docs/experiments/003-selfplay-v2-baseline-mix-0.5.md).

## How to test

- **Metric:** whether a pool's `checkpoints()` listing includes any file
  from a run other than the current one.
- **Gate:** a run-scoped pool (`run_id` set) lists only checkpoints
  written under its own `checkpoint_dir`; an unscoped pool over the same
  tree still sees the stale file, confirming the difference is the
  scoping, not an unrelated environment change.
- **Cost estimate:** near zero for a newly-started run using
  `run_id` — the fix is additive and keyword-only. Retrofitting an
  in-flight consumer requires moving existing checkpoints into a
  subdirectory or switching to a `run_id`-scoped pattern.

## Result

Fixed and shipped: gamekit PR #24, merged 2026-09-20, `OpponentPool` gains
a keyword-only `run_id` that scopes the glob to `directory / run_id`, plus
a `checkpoint_dir` property and a public `checkpoints()` method — released
as gamekit 0.3.0. Closes gamekit issue #23.

**Tested by:** truco-py logs
[004](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/004-april-selfplay-collapse.md)
and
[005](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/005-june-threshold-mix-collapse.md)
recorded the failure (truco-py has not yet adopted the fix — pinned at
gamekit 0.2.0 as of those logs); catan log
[003](https://github.com/guidodinello/catan/blob/main/docs/experiments/003-selfplay-v2-baseline-mix-0.5.md)
used the fix from the start and never reproduced the collapse.

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [011 — KL guard vs the previous snapshot](011-kl-guard.md)
