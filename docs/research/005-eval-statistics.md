# 005 — Eval statistics: Wilson intervals and eval-in-loop

**Status:** validated
**Last touched:** 2026-09-30

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

**Tested by:** catan log
[001](https://github.com/guidodinello/catan/blob/main/docs/experiments/001-ppo-vs-random.md)
is the source of the 81.75% eval-in-loop gate result above.

### In-loop eval design (2026-09-29, catan PR #27)

catan issue [#25](https://github.com/guidodinello/catan/issues/25) found
that `rl/train.py` passed the same `seed=cfg.seed + 977` at every in-loop
eval, so all of them replayed the same 200 game setups. catan
[PR #27](https://github.com/guidodinello/catan/pull/27) (merged as
`95c59d8`, closes #25) fixed it by splitting the in-loop eval in two;
the design and its caveats are written up in catan's
[`docs/experiments/README.md`](https://github.com/guidodinello/catan/blob/main/docs/experiments/README.md#in-loop-eval-caveats).
No external source; observed in catan PR #27.

- **Fixed paired set drives only the stop.** The guard eval keeps
  `cfg.seed + 977` (unchanged, so comparable with catan 004/006) and is
  the only input to `RegressionGuard`'s stop decision. A fixed game set
  makes the checkpoint-to-checkpoint comparison paired, which is what a
  stop-on-regression rule wants.
- **Fresh seed picks the best checkpoint.** A second n=200 eval at
  `cfg.seed + 977 + step` (`step` is cumulative, so a `--resume` leg never
  replays the previous leg's seeds) is the *reported* rate and alone
  picks `best_checkpoint` / `best_win_rate`. Both rates are logged, as
  `reported=` and `guard(fixed)=`.
- **Why the guard stays fixed (upper bound).** The PR's planner simulated
  the guard in pure Python with a flat 21% policy and independent n=200
  draws (margin 0.10, patience 2, seeded at 16.2%): about 24% of 40-eval
  runs and about 64% of 93-eval runs stop falsely. That is an *upper
  bound* for the fixed set, because pairing is only partial: dice and
  steals draw from the shared `state.rng`, so they drift once policies
  differ, while board, dev deck, starting player, seat and opponent
  seeds stay fixed. Re-running the simulation for this note (5,000 runs
  per length) gave 25% and 64%, within simulation noise of those figures.
- **Cost.** About 31 s per n=200 eval (catan log 006: 20.4 min over 40
  evals against 185.7 min of training), so the second eval adds about
  20 min to a 10M-step run (about 4% of an 8 h budget), or about 9% fewer
  training steps on a run bound by a timeout.
- **What it does not fix.** Catan experiments 001-006 used the fixed seed
  for everything, including best-checkpoint selection. The new scheme
  removes the replay of one sample, but the best reported rate is still
  a maximum over about 40 noisy n=200 evals, so it stays biased upward.
  **n=4000 confirmation of the chosen checkpoint remains mandatory.**
  See [009](009-longer-runs-and-resume.md) for how this bears on the
  winner's-curse gap.

**Linked from** three truco-py logs, none of which confirm this note —
each is recorded here for the numbers, by its own stated verdict, not as
independent tests of the hypothesis:

- log [001](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/001-mc-threshold-derivation.md)
  ("does not test this note's hypothesis" — its MC simulations were
  run to completion at a fixed `n`, not watched in-loop for early
  collapse, which is the concern this note is actually about).
- log [006](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/006-seat-rotation-deconfound.md)
  ("inconclusive on its own terms" — adopted `wilson_interval` for the MC
  experiments, but the seat-rotation fix itself is a methodology change,
  not a statistics one, and no post-fix RL win rate was computed at a
  large enough `n` to confirm or refute anything with the new interval
  method; see [016](016-positional-advantage-rotation.md)).
- log [007](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/007-gamekit-rl-adapter-parity.md)
  ("does not test this note's hypothesis" — a determinism/regression
  parity check for a refactor, not an experiment measuring whether
  Wilson intervals or eval-in-loop monitoring improve anything; it is,
  incidentally, the first post-rotation RL benchmark with a computed
  Wilson interval, 79.0% [72.8%, 84.1%] vs `ThresholdAgent`, n=200).

Catan log
[007](https://github.com/guidodinello/catan/blob/main/docs/experiments/007-human-games.md)
also cites this note, for the Wilson intervals on its 12-game arms. It
does not test this note; the note it tests is
[018](018-human-baseline-sanity-check.md).

### Paired fork statistic (2026-09-30, catan log 009 / PR #43)

When the question is what *one decision* does to the win rate, two
independent n=4000 arms are the wrong tool. catan log
[009](https://github.com/guidodinello/catan/blob/main/docs/experiments/009-critic-calibration.md) (catan [PR #43](https://github.com/guidodinello/catan/pull/43)) measured it with a paired fork, and this
records the statistic so other logs can reuse it. No external source;
observed in that log.

- **Design.** At a sampled decision point (one per game, so the units are
  independent), copy the state and the agents, then play the game on from
  both branches (accept vs reject). Repeat for K pairs (K=16). Both
  branches of pair k are reseeded with the same RNG seed (common random
  numbers), so dice, draws and steals start identical and differ only
  through the decision. The per-decision effect is
  `dwin = mean over k of (win_accept - win_reject)`.
- **Inference.** The unit is the decision (game), not the playout. Report
  the mean of `dwin`, or a rank correlation of `dwin` with a predictor,
  with a **percentile bootstrap of the paired differences** (log 009: B =
  2000 over 3856 decisions). Do **not** use `gamekit.mc.two_proportion_test`
  on the two branches' win counts: the branches share their random numbers
  and starting state, so they are not independent samples, which is
  also what the benchmark log's "seats of one role are dependent" caveat
  warns about. The gate here is the CI of the *difference* excluding 0,
  not non-overlapping arm intervals.
- **Scale.** In log 009 the per-decision `dwin` has SD 0.103 at K=16; the
  mean over 3856 decisions has SE about 0.0017, so the design resolves
  differences of roughly 0.3-0.6 points and no smaller.
- **Caveats.** Common random numbers keep the branches coupled only until
  their decisions consume the RNG differently, so the variance saved was
  not measured against independent seeds. A "reject" branch that lets
  later responders act is not the same as withdrawing the offer; say
  which one the fork uses.

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [006 — Uniform-over-atoms baseline is not `RandomAgent`](006-uniform-atoms-baseline.md)
- [009 — Longer runs / resume when the curve has not bent](009-longer-runs-and-resume.md)
- [011 — KL guard vs the previous snapshot](011-kl-guard.md)
- [016 — Positional (mano) advantage must be rotated out of a benchmark arm](016-positional-advantage-rotation.md)
- [018 — Human baseline as a sanity check for learned / hand-written agents](018-human-baseline-sanity-check.md)
- [020 — Modular agent: separate trade module over a strategy policy](020-modular-trade-agent.md)
