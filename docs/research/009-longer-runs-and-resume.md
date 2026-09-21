# 009 — Longer runs / resume when the curve has not bent

**Status:** idea
**Last touched:** 2026-09-20

## Hypothesis

When a training curve is still climbing and hasn't plateaued, continuing
training (or resuming from the last checkpoint for another budget) should
keep improving the win rate, rather than the gains having already
saturated within the budget tried so far.

## Why we believe it

- No external source for this specific claim yet — it is an open question,
  not something either consumer repo has tested. catan's PR #18 run
  **stopped early on purpose**, not because the curve had bent: "the
  smoke run (8k steps) already showed learning (49%→70% at 60k/120k
  steps), and the first two real checkpoints were stable and decisive
  (87.5% [82.2%, 91.4%] at 500k, 84.0% [78.3%, 88.4%] at 1M — overlapping
  CIs, not a fluke). Spending the rest of the 8h budget on a gate already
  this clear would have been waste." — no external source; observed in
  catan PR #18's body.
- truco-py's threshold-training runs, by contrast, plateaued and did *not*
  keep improving with more steps: "Threshold training plateaued at ~84%
  between 2M and 5M steps. More threshold training will not push past
  this." — no external source; observed in truco-py
  `docs/session-2026-06-06.md`.

## Why this is still `idea`, not `planned` or `rejected`

catan's gate was cleared early *by choice*, so "does training longer keep
helping" was never actually tested there — it's untested, not rejected.
truco-py's plateau is real evidence of a ceiling, but for a *different*
training regime (pure threshold self-play, not the gate-vs-heuristic setup
[001](001-self-play-opponent-mix.md) is tracking), so it doesn't settle
the question for catan's Phase 5 either.

## How to test

- **Metric:** win rate vs the fixed gate opponent, checkpointed at regular
  intervals well past the point a prior run stopped.
- **Gate:** a materially higher win rate (non-overlapping CI) at a later
  checkpoint than the one a prior run stopped at, for the same
  hyperparameters.
- **Cost estimate:** the cost of one extended run — potentially the full
  training budget a prior run left unused, plus in-loop eval (see
  [005](005-eval-statistics.md)) so a resumed run that starts collapsing
  is caught rather than wasted.

## Result

Not yet attempted for the gate-vs-heuristic setup this note is about.

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
