# 006 — Uniform-over-atoms baseline is not `RandomAgent`

**Status:** validated
**Last touched:** 2026-09-20

## Hypothesis

On a factored/sequential action space, sampling atoms uniformly is *not*
the same distribution as `RandomAgent` (uniform over legal actions) — so
"beats 25% (the fair 4-player seat share)" is a weaker claim than it looks
for a policy that hasn't learned anything yet.

## Why we believe it

- Uniform over atoms weights action *kinds* roughly evenly rather than
  weighting each concrete action instance evenly, which behaves much more
  like a stratified-random baseline than a flat-random one — no external
  source; observed in catan `rl/evaluate.py:177-190`'s `random_predictor`
  docstring (PR #18).
- Measured directly: uniform-atom play scored **32.3% [27.3%, 37.8%]**
  against three `RandomAgent`s but only **24.0% [19.5%, 29.1%]** against
  three `StratifiedRandomAgent`s, at n=300 — no external source; observed
  in catan `rl/evaluate.py:177-190` and pinned as
  `tests/rl/test_training_wiring.py:118`
  (`test_uniform_atom_play_matches_stratified_random_not_flat_random`).

## What this does *not* show

This is not evidence that an untrained policy is "good" — it's the
opposite: it shows the gate needs to be set relative to this baseline
(32.3% vs flat-random opponents), not relative to the naive 25% fair-seat
share, or a policy that has learned nothing at all can already look like
it cleared the bar. On a factored action space, "beats 25%" alone is a
weak claim and should not be used as a training-progress gate by itself
— it's a property of the action-space parameterization, not of the
policy.

## How to test

- **Metric:** win rate of a uniform-atom baseline vs both `RandomAgent`
  and `StratifiedRandomAgent`, at a fixed `n`.
- **Gate:** none — this note documents a baseline calibration finding, not
  a technique with its own pass/fail bar. Its output feeds the gates in
  [001](001-self-play-opponent-mix.md) and [005](005-eval-statistics.md).
- **Cost estimate:** cheap — no training required, just running the
  uniform-atom policy through the existing eval harness.

## Result

catan PR #18, n=300, pinned as a regression test (see "Why we believe it"
above). No further experiment planned; this is a calibration fact used by
other notes' gates.

## Related notes

- [004 — Factored action head via sequential atom composition](004-factored-action-head.md)
- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
