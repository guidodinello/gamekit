# 005 — Eval statistics: Wilson intervals and eval-in-loop

**Status:** validated
**Last touched:** 2026-09-20

## Hypothesis

A win-rate point estimate is not a result on its own — it needs a
confidence interval computed with a method that doesn't misbehave at
extreme proportions, and it needs to be watched *during* training rather
than only checked at the end, or a collapse burns the whole run budget
before anyone notices.

## Why we believe it

- The Wald (normal-approximation) interval for a binomial proportion has
  "chaotic coverage properties" that are "far more persistent than
  previously appreciated", especially near 0 or 1 or at small `n`; Wilson
  (or Agresti-Coull at larger `n`) is the recommended replacement —
  [Brown, Cai & DasGupta 2001, *Interval Estimation for a Binomial
  Proportion*, Statistical Science
  16(2)](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full)
  — takeaway: this isn't a style preference, the naive interval is
  measurably wrong in exactly the small-`n`/extreme-`p` regime a win-rate
  eval lives in.
- `gamekit.mc.wilson_interval` already implements this and is used by
  `gamekit.benchmark`'s win-rate summaries (`src/gamekit/mc/
  intervals.py:55`, `src/gamekit/benchmark.py:58,91`) — the eval-statistics
  discipline this note argues for is already load-bearing infrastructure,
  not a proposal.
- At n=200, `gamekit.mc.wilson_interval` gives `[5.4%, 13.2%]` for a 17/200
  win rate (8.5%) and `[6.6%, 14.9%]` for a 20/200 win rate (10%) — the two
  intervals overlap almost entirely, too coarse to distinguish an 8.5% win
  rate from a 10% one, which is exactly the gap [001](001-self-play-opponent-mix.md)
  needs to resolve — no external source; observed by running
  `wilson_interval(17, 200)` / `wilson_interval(20, 200)` against
  `gamekit.mc`.
- catan's `rl/evaluate.py` builds exactly this discipline into training:
  `WinRate` + a Wilson-interval `summarize()` used both by an in-loop eval
  callback between training chunks and standalone, with `summarize(0, 0)`
  raising rather than silently reporting a 0% rate — no external source;
  observed in catan `rl/evaluate.py` (PR #18).
- truco-py is the negative case that motivates watching the curve, not
  just computing a CI at the end: it had **no** eval-in-loop and **no**
  win-rate confidence interval anywhere — every headline RL number (85.3%,
  84.0%, 56.0%, 57.3%) is a bare point estimate, evaluated only after the
  fact at n=150 or smaller. Its only automated stopping rule was a manual
  "kill criterion: if 3 consecutive checkpoints &lt; 80%, stop" — no
  margin, no patience, no automated evaluator. Two of its self-play runs
  collapsed before anyone was watching closely enough to catch it early,
  and its own numbers show the CI problem directly: the "prior best"
  moved from 85.3% to 84.0% purely from a different seed at n=150, a gap
  well inside what a Wilson interval at that sample size would call noise
  — no external source; observed in truco-py `docs/session-2026-06-05.md`,
  `docs/session-2026-06-06.md`.

## How to test

- **Metric:** report every win rate with its Wilson CI, always.
- **Gate:** a claimed improvement must have non-overlapping CIs against
  the baseline it's compared to, at the sample size actually used.
- **Cost estimate:** near zero — `wilson_interval` is already in
  `gamekit.mc`; the cost is discipline (always call it), plus wiring an
  eval-in-loop callback into the training loop once per game.

## Result

catan PR #18 built `rl/evaluate.py`'s eval-in-loop + Wilson-CI reporting
and used it for the 81.75% [80.5%, 82.9%] gate result (n=4000, see
[003](003-discount-horizon.md)). truco-py never adopted eval-in-loop or CI
reporting for RL evals, and paid for it in undetected collapses (see
[001](001-self-play-opponent-mix.md)'s counter-evidence and
[011](011-kl-guard.md)).

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [006 — Uniform-over-atoms baseline is not `RandomAgent`](006-uniform-atoms-baseline.md)
- [011 — KL guard vs the previous snapshot](011-kl-guard.md)
