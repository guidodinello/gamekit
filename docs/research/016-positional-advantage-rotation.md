# 016 — Positional (mano) advantage must be rotated out of a benchmark arm

**Status:** validated for the mechanism; inconclusive for the effect size
**Last touched:** 2026-09-21

## Hypothesis

If a benchmark's engine always seats the same role first (or gives it ties
in the same direction), every win rate it produces carries a positional
advantage confound, independent of the agents' actual skill. Rotating
which role occupies which seat across the arm removes the confound; not
rotating it does not.

## Why we believe it

- truco-py's engine defaulted `current_player=0` at game construction and,
  before the fix, always started Team A and gave Team A ties — a
  permanent mano (first-to-act) advantage for one team — no external
  source; observed in truco-py `engine/game.py:79` and truco-py#1's PR
  description.
- gamekit's own benchmark infrastructure already treats this as mandatory,
  not optional: "Seat rotation is mandatory, not a refinement... role-to-
  seat assignment rotates so each role occupies each seat an equal number
  of times across an arm" — no external source; observed in gamekit
  `src/gamekit/benchmark.py:16-19` and `src/gamekit/seats.py`, which
  implement `rotate()` and `seat_occupancy_counts()` as the load-bearing
  fairness proof for `run_arm`.
- The general pattern (first-move/seat advantage requiring deliberate
  handling, not being ignorable) is documented for self-play training and
  evaluation broadly: "in TTGs the initial player often has an advantage
  or disadvantage, or some asymmetry. For self-play and later for
  evaluation it is crucial to train agents that can play well from any
  player position" — [Balla, Long, Goodman, Gaina & Perez-Liebana 2024,
  *PyTAG: Tabletop Games for Multi-Agent Reinforcement
  Learning*](https://arxiv.org/html/2405.18123v1) — takeaway: this is a
  known, named failure mode across tabletop-game RL, not a truco-specific
  quirk.
- catan's own gate benchmarks are seat-rotated by construction, not as an
  afterthought — no external source; observed in catan logs
  [001](https://github.com/guidodinello/catan/blob/main/docs/experiments/001-ppo-vs-random.md)
  ("seat-rotated (z=-95.7 vs the 25% null, p≈0)") and
  [003](https://github.com/guidodinello/catan/blob/main/docs/experiments/003-selfplay-v2-baseline-mix-0.5.md)
  ("No authoritative (n=4000, seat-rotated) benchmark was run for v1").

## Measured effect size — real at the code level, inconclusive at n=1000

truco-py#1 measured the de-confounding shift directly: `threshold_vs_random`
at `seed=42, n=1000`, comparing rotation pinned to zero (reproducing the
pre-fix behavior) against production rotation:

- Pinned-zero (pre-rotation-equivalent): 476/462/62 win/loss/tie →
  **47.6%, [44.5%, 50.7%]**
- Production (post-rotation): 456/473/71 →
  **45.6%, [42.5%, 48.7%]**

The shift is in the expected direction (Threshold's edge shrinks once it
no longer always gets the mano-advantaged seat), and the mechanism is
confirmed at the source-code level (`current_player=0` was a real,
permanent seat assignment, not a hypothesized one). But **the two
intervals overlap heavily** — at n=1000 this specific measurement cannot
distinguish the de-confounding shift from noise. Say both things plainly:
the mechanism is validated (real, present in the code, fixed by a real
change), the *size* of its effect on this particular matchup is not
resolved by this sample.

A second consequence worth stating explicitly: every truco-py win rate
recorded before 2026-09-20 — every RL and rule-based number in this
repo's history — was measured under the pre-rotation, tie-favors-Team-A
rule and is not directly comparable to anything measured after it.

## How to test

- **Metric:** win rate for the same matchup, measured with rotation pinned
  to reproduce the old behavior vs. with production rotation, same seed.
- **Gate:** non-overlapping Wilson CIs at whatever `n` the claim needs to
  support — n=1000 was not enough here.
- **Cost estimate:** the benchmark run itself is cheap (a few seconds to
  minutes per arm); the cost is in re-running every stale pre-rotation
  benchmark that a project wants to make comparable to future numbers.

## Result

**Tested by:** truco-py log
[006](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/006-seat-rotation-deconfound.md)
measured the 47.6%→45.6% shift at n=1000 (mechanism confirmed, effect size
inconclusive at that sample). No post-rotation RL benchmark at a larger
`n` exists yet to resolve the effect size for an RL-vs-scripted matchup
specifically — truco-py log
[007](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/007-gamekit-rl-adapter-parity.md)
is post-rotation but was a regression/parity check for a refactor, not
designed to isolate the rotation effect.

## Related notes

- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
