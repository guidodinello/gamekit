# 003 — Discount horizon vs episode length

**Status:** validated
**Last touched:** 2026-09-20

## Hypothesis

The discount factor `gamma` must be chosen relative to how many learner
decisions an episode actually contains — copying a value tuned for a
short-horizon game onto a long-horizon one makes a terminal reward signal
effectively invisible at the opening of the episode.

## Why we believe it

- Potential-based reward shaping, `F(s,a,s') = gamma * Φ(s') - Φ(s)`, is
  provably policy-invariant — it changes how fast the signal arrives, not
  which policy is optimal — [Ng, Harada & Russell 1999, *Policy Invariance
  Under Reward Transformations: Theory and Application to Reward
  Shaping*](https://dl.acm.org/doi/10.5555/645528.657613) — takeaway:
  shaping is safe to combine with a high `gamma` precisely because it
  can't distort the optimum, only the learning speed.
- truco-py never examined this: its `gamma=0.99`/`gae_lambda=0.95` are
  unremarked constants in `training/train.py`, with no rationale recorded
  anywhere in its docs and no alternative ever discussed — no external
  source; observed in truco-py `training/train.py:400-401` (absence of
  discussion is itself the finding: a short-episode game (30–80 steps per
  hand) can get away with an unexamined default that a longer-episode game
  cannot).
- catan measured ~450 learner decisions per episode (4-player games) and
  set `gamma=0.999` specifically because `gamma=0.99` (copied from
  truco-py) would put the effective horizon around 100 steps — leaving a
  terminal win/loss signal nearly invisible at the opening placements — no
  external source; observed in catan `rl/train.py:8-14` and
  `rl/reward.py:31-47` (PR #17/#18).

## How to test

- **Metric:** win rate vs a fixed opponent, measured across training, at a
  given `gamma`.
- **Gate:** the chosen `gamma` should clear the same gate a lower `gamma`
  fails to clear at a comparable step budget, or reach it measurably
  faster.
- **Cost estimate:** cheap relative to the full training run — `gamma` is
  a single config value, no code change once shaping is already
  potential-based.

## Result

catan PR #18: **81.75% [80.5%, 82.9%] win rate vs 3 `RandomAgent`s, 4p,
n=4000**, seat-rotated (z=-95.7 vs the 25% null, p≈0), with
`gamma=0.999`. Result committed at
`experiments/results/benchmark_rl_vs_random_p4.json` in the `catan` repo.
The gate ("&gt;25% with the Wilson CI excluding it") was cleared by 55
points, not marginally.

Note on episode-step reporting: catan's own sources disagree on the raw
engine-step count per episode (1584 in PR #17's body, ~1580 in
`rl/train.py:9`, "roughly 1200" in `rl/reward.py:41`) — only the ~450
*learner-decision* figure is consistent across all three, and that is the
number `gamma=0.999` is actually chosen against.

## Related notes

- [002 — Behavior-cloning warm start before PPO](002-bc-warm-start.md)
