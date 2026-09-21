# 011 — KL guard vs the previous snapshot

**Status:** idea
**Last touched:** 2026-09-20

## Hypothesis

An automated guard comparing the current policy against a recent snapshot
(by KL divergence, or by the entropy/clip-fraction signals that
accompanied a real collapse) could catch a self-play collapse in progress
and stop or roll back training before the run's time budget is wasted —
but the guard has to watch for the collapse that actually happened, not
the one PPO's own tuning guidance warns about.

## Why we believe it

- PPO's own adaptive-KL-penalty variant reduces the penalty coefficient
  when divergence between old and new policy is *below* a target and
  increases it when *above* — a mechanism built around watching KL move
  in either direction — [Schulman et al. 2017, *Proximal Policy
  Optimization Algorithms*](https://arxiv.org/abs/1707.06347) — takeaway:
  treating KL divergence as a signal worth actively watching (not just
  logging) is an established PPO pattern, even though Schulman et al.
  found the *clipped* objective outperforms the KL-penalty variant as the
  optimization method itself.
- truco-py's collapse is the concrete motivating case, and its actual
  signature is the **opposite** of a KL spike: `approx_kl` went to `0.0`
  and `clip_fraction` to `0` as the policy froze into a degenerate
  loop (`ep_len_mean` 2 → 165, `ep_rew_mean` 0.4 → -7.28,
  `entropy_loss` -0.14 → -1.18e-05) — no external source; observed in
  truco-py `logs/train_selfplay_gpu.log` and `docs/session-2026-06-06.md`.
  A guard that only fires on a KL *increase* would have missed this
  collapse entirely — it needs to watch for the policy going flat
  (entropy/KL/clip-fraction all collapsing toward zero together), not only
  for divergence blowing up.

## How to test

- **Metric:** `approx_kl`, `clip_fraction`, and `entropy_loss` tracked
  per training iteration against a rolling recent-snapshot baseline.
- **Gate:** the guard fires (pauses or rolls back training) on the
  truco-py collapse signature — all three going toward zero together —
  within some bounded number of iterations of it starting, without firing
  on healthy runs (e.g. the stable metrics reported for catan's v1
  self-play run in [001](001-self-play-opponent-mix.md):
  `entropy_loss -0.29 → -0.17`, `approx_kl 0.002–0.004` — no external
  source; observed in the 2026-09-20 self-play v1 run, catan branch
  `phase5-rl-selfplay` (unmerged); these are not from a merged catan PR).
- **Cost estimate:** low to implement (a callback reading metrics SB3
  already reports), needs at least one intentionally-triggered collapse
  run (or a replay of truco-py's logged one) to validate the trigger
  condition before trusting it on a real run.

## Result

Not yet implemented in either consumer repo. This is a response to
[005](005-eval-statistics.md)'s finding that truco-py's only stopping rule
was a manual, coarse "3 consecutive checkpoints < 80%" check.

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
- [010 — Entropy schedule instead of a fixed coefficient](010-entropy-schedule.md)
