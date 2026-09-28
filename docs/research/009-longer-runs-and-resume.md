# 009 — Longer runs / resume when the curve has not bent

**Status:** planned
**Last touched:** 2026-09-27

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
  `docs/session-2026-06-06.md`. **This claim is weaker than it looks**:
  truco-py log
  [003](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/003-threshold-training-plateau.md)
  shows the 5M-step checkpoint's benchmark was n=50, whose Wilson interval
  ([71.5%, 91.7%]) cannot exclude the repo's own 90% goal — at that sample
  size the data cannot distinguish a genuine plateau from a checkpoint
  that would have crossed 90% with a larger benchmark. So this counts as
  a real, but statistically underpowered, ceiling observation.

## Why this was `idea`, and what moved it to `planned`

catan's gate was cleared early *by choice*, so "does training longer keep
helping" was never actually tested there — it's untested, not rejected.
truco-py's plateau is real evidence of a ceiling, but for a *different*
training regime (pure threshold self-play, not the gate-vs-heuristic setup
[001](001-self-play-opponent-mix.md) is tracking), so it doesn't settle
the question for catan's Phase 5 either.

The 2026-09-27 refinement below turned it into a concrete, pre-registered
run. It also changed what the question is.

## Refinement (2026-09-27)

- **The "oscillation" in both catan logs is not distinguishable from a
  flat curve.** A chi-square homogeneity test over the in-loop evals
  (n=200 each) finds no evidence the win rate moved at all:
  catan log 004's twelve evals from 250k to 3M (10.5–20.0%, pooled 14.5%)
  give χ²=11.7, df=11, p=0.39; catan log 003's last fifteen evals from
  4.0M to 7.5M (10.5–15.5%, pooled 13.0%) give χ²=6.1, df=14, p=0.96 — no
  external source; computed from the tables in those two logs. So "has
  the curve bent?" cannot be read off the in-loop eval at all: at n=200
  the minimum detectable difference between two evals (80% power,
  α=0.05) is about **10 points**, larger than the whole band. At n=4000
  it is about **2.3 points**.
- **Consequence for the design:** the in-loop eval stays a collapse alarm
  ([005](005-eval-statistics.md)), not the measurement. The measurement
  is a small number of **fixed, pre-chosen** checkpoints, each
  benchmarked at n=4000. Picking the best in-loop checkpoint and then
  benchmarking it has a winner's-curse bias, and log 004 shows it: its
  20.0% in-loop peak benchmarked at 16.2% [15.09%, 17.37%].
- **Two gaps in catan's `--resume` path that a multi-leg run hits**
  (catan `rl/train.py` at the time of writing):
  1. `RegressionGuard` starts from `best_rate=-1.0` on every invocation.
     A resumed leg can't see a regression relative to the previous leg's
     peak — the known limitation in catan log 003's follow-up section.
     `--bc-init-rate` already solves this for `--bc-init`, but not for
     `--resume`.
  2. `--resume` loads the pickled `learning_rate`/`ent_coef` and ignores
     `--learning-rate`/`--ent-coef` without saying so (only the `--bc-init`
     path passes `custom_objects`). For this note that's what we want,
     since the run must keep the same hyperparameters. It is a trap for
     [010](010-entropy-schedule.md), which needs to change `ent_coef` on
     a resumed run.

## How to test

**Prerequisites (catan, small):**

- Add `--resume-rate` / `--resume-best` (or generalize `--bc-init-rate`
  to both warm-start paths), so the resumed leg's `RegressionGuard` is
  seeded with the previous leg's n=4000 rate and checkpoint rather than
  `-1.0`.
- Log the effective `ent_coef`/`learning_rate` after `MaskablePPO.load`
  on the resume path, and fail if `--ent-coef`/`--learning-rate` were
  passed explicitly with values that differ from the pickled ones. That
  way a resume can't silently run different hyperparameters than the
  command line says.

**Protocol (catan, current best config):**

- **Start:** `rl_runs/selfplay/catan_bc_ft/catan_bc_ft_final.zip` (the end
  of catan log 004's 3M-step fine-tune). Same label, pool and flags as
  log 004 (`--baseline-mix 0.5`, `--envs 8`, `--seed 3`, `--eval-opponents
  heuristic`, `--eval-every 250000`, `--eval-episodes 200`). No
  `--ent-coef`/`--learning-rate`: the pickled 0.01/1e-4 carry over.
  Guard seeded with 16.2% (log 004's n=4000 result for its 2M
  checkpoint).
- **Length:** `--steps 3000000`, which takes the run from 3M to 6M
  cumulative (the same as a second leg of log 004).
- **Measurement points (fixed before the run):** the checkpoints at 3M
  (the resume start, which log 004 never benchmarked at n=4000), 4M, 5M
  and 6M. Each is benchmarked vs 3 `HeuristicAgent` with the log 004
  protocol (n=4000, `engine_seed_base=1`, `driver_seed_base=1`,
  seat-rotated, `deterministic=True`).

- **Metric:** n=4000 win rate vs 3 `HeuristicAgent` at 3M/4M/5M/6M
  cumulative steps, with Wilson intervals.
- **Gate (pre-registered):**
  - *validated* if the 6M checkpoint beats the 3M checkpoint by
    `two_proportion_test` p < 0.05 (`gamekit.mc.testing`), **and** the
    4M→5M→6M points don't go down (the gain is a trend, not a single
    lucky point).
  - *rejected* for this config if the 6M-vs-3M difference is not
    significant and its point estimate is under +2.3 points (the n=4000
    MDE), meaning the curve has bent within the precision this protocol
    can resolve.
  - *inconclusive* otherwise (e.g. significant but not monotone). The
    next step is then a 6M→9M leg, not a new note.
  - Comparing against 2M (16.2%) as well is fine as a secondary
    number. It's not the gate, because 2M was chosen as the in-loop
    peak and carries the winner's-curse bias described above.
- **Cost estimate:** ~90 min of training (log 004 ran 3M steps at
  `--envs 8` in ~90 min, including eval) plus four n=4000 benchmarks.
  The benchmark step dominates, so use as many `--workers` as the box
  allows. The prerequisites are about half a day of catan work,
  including tests for the seeded guard.

**Optional, cheap (truco-py):** re-benchmark the 2M and 5M
threshold-training checkpoints from truco-py log 003 at n≥1000 each, if
they still exist on disk. That alone resolves whether truco-py's "plateau
at ~84%" is real or an artifact of n=50, with no new training.

## Result

Not yet attempted for the gate-vs-heuristic setup this note is about.

**Linked from** truco-py log
[003](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/003-threshold-training-plateau.md)
(verdict: *inconclusive* — "exactly the open question gamekit#009
records," per the log's own wording, for the underpowered-benchmark
reason described above). catan logs
[001](https://github.com/guidodinello/catan/blob/main/docs/experiments/001-ppo-vs-random.md)
and
[003](https://github.com/guidodinello/catan/blob/main/docs/experiments/003-selfplay-v2-baseline-mix-0.5.md)
both name this note as their own open question rather than testing it:
log 001 stopped early by choice with a still-decisive gate; log 003's
in-loop oscillation (10.5-15.5% across the last ~2.5M steps) "shows a real
ceiling for this configuration within the budget used, not a
still-climbing curve" but explicitly leaves open "whether more steps at
the same settings would eventually break out."

**Motivated by** catan log
[004](https://github.com/guidodinello/catan/blob/main/docs/experiments/004-bc-warm-start.md),
whose BC-warm-start fine-tune oscillated in a 10.5-20.0% band vs
`HeuristicAgent` for its entire 3,000,000-step run without `RegressionGuard`
ever triggering a stop — every dip stayed within its margin of the running
best, so the run's own follow-up section names this note's question
directly: "would more fine-tune steps past 3M keep climbing, given the run
never regressed enough to stop?" Not tested — log 004 stopped at its
pre-set step ceiling, not because the curve had bent.

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [002 — Behavior-cloning warm start before PPO](002-bc-warm-start.md)
- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
- [010 — Entropy schedule instead of a fixed coefficient](010-entropy-schedule.md)
- [011 — KL guard vs the previous snapshot](011-kl-guard.md)
