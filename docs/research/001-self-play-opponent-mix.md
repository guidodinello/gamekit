# 001 — Self-play opponent mix vs a fixed baseline

**Status:** running
**Last touched:** 2026-09-21

## Hypothesis

When the eval gate is win rate against a fixed baseline agent, giving that
baseline a heavier fixed share of the self-play opponent pool speeds up
progress against it, more than sampling almost entirely from past
checkpoints does.

## Why we believe it

- Naive self-play (playing only the current policy) causes "forgetting"
  and rock-paper-scissors cycling — an agent that beats itself can still
  lose to an earlier version, or to a fixed baseline it has stopped
  practicing against — [AlphaStar, Vinyals et al. 2019, *Grandmaster level
  in StarCraft II using multi-agent reinforcement
  learning*](https://www.nature.com/articles/s41586-019-1724-z) —
  takeaway: mixing in older/fixed opponents ("fictitious self-play —
  playing against a mixture of all previous strategies") is the fix, not
  an optional extra.
- OpenAI Five trained rollout workers with a fixed split — "play the
  latest policy against itself for 80% of games, and play against older
  policies for 20% of games" — [*Dota 2 with Large Scale Deep
  Reinforcement Learning*, arXiv:1912.06680](https://arxiv.org/abs/1912.06680)
  — takeaway: a *nonzero, fixed* non-self-play fraction is a standard
  knob, not a hack specific to one game.
- The general theory behind mixing in past strategies rather than only the
  current one is fictitious play in extensive-form games —
  [Heinrich, Lanctot & Silver 2015, *Fictitious Self-Play in
  Extensive-Form Games*](http://proceedings.mlr.press/v37/heinrich15.pdf)
  — takeaway: convergence arguments for self-play assume a mixture over
  past policies, not just the latest one.
- catan v1 (`baseline_mix 0.2`) plateaued at ~8.5% vs `HeuristicAgent`
  with healthy PPO health metrics (entropy_loss -0.29 → -0.17, approx_kl
  0.002–0.004, explained_variance 0.79–0.91 — training was stable, it just
  wasn't winning); v2 (`baseline_mix 0.5`, same seed checkpoint) climbed
  2.5% → 15.0% by 3.25M steps — no external source; observed in the
  2026-09-20 self-play v1/v2 runs, catan branch `phase5-rl-selfplay`
  (unmerged; not yet a PR).

## Counter-evidence (do not ignore)

truco-py ran the opposite experiment and got the opposite lesson. Its
self-play pool collapsed **at both `threshold_mix=0.2` and
`threshold_mix=0.5`** — the April run at the default 0.2 regressed to
57.3% vs `ThresholdAgent` (down from 85.3%), and raising the mix to 0.5 on
2026-06-06 collapsed again a second time (`ep_rew_mean` 0.4 → -7.28,
`ep_len_mean` 2 → 165 over ~11 minutes of wall clock). **Root cause,
confirmed by gamekit#24 (see [013](013-selfplay-pool-contamination.md)):**
`OpponentPool` sampled uniformly over every file matching the pool's glob
in a shared checkpoint directory, so a *stale, already-collapsed*
checkpoint from an *earlier run* (the April run's own
`truco_selfplay_final.zip`) stayed eligible for the June run too and
contaminated it — a related but distinct mechanism from what was
originally hypothesized here at the time (that all slots resolved to the
same *latest* checkpoint within one run); the actual bug was cross-run
staleness, not within-run degeneracy — no external source; observed in
truco-py logs
[004](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/004-april-selfplay-collapse.md)/[005](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/005-june-threshold-mix-collapse.md)
and gamekit PR #24.

So a heavier baseline share is not sufficient on its own to prevent
collapse; the mix ratio matters less than whether something is actually
*watching* the eval curve and stopping before a collapse burns hours, and
less than whether the pool itself is free of stale cross-run checkpoints
([013](013-selfplay-pool-contamination.md)). This is the conclusion
catan's own `rl/train.py` self-play docstring reaches: "the floor's exact
value matters less than having `RegressionGuard` ... actually watching the
eval curve and stopping before hours are wasted past a collapse, which is
what neither of truco's runs had" — no external source; observed in catan
`rl/train.py:22-31` (uncommitted).

## How to test

- **Metric:** win rate vs `HeuristicAgent`, Wilson 95% CI (see
  [005](005-eval-statistics.md)).
- **Gate:** >25% seat share with the CI excluding it, matching the
  `rl_vs_random` gate's shape from catan PR #17/#18.
- **Cost estimate:** one committed training run per mix value, evaluated
  in-loop rather than only at the end (see [005](005-eval-statistics.md),
  [011](011-kl-guard.md)) so a collapse is caught rather than burning the
  full budget.

## Result

**Tested by:** catan logs
[002](https://github.com/guidodinello/catan/blob/main/docs/experiments/002-selfplay-v1-baseline-mix-0.2.md)
(`baseline_mix=0.2`, rejected — plateaued at 2.5-9% vs `HeuristicAgent`
through 3.7M steps, healthy training dynamics, no collapse) and
[003](https://github.com/guidodinello/catan/blob/main/docs/experiments/003-selfplay-v2-baseline-mix-0.5.md)
(`baseline_mix=0.5`, learning confirmed but this note's own gate not met —
authoritative n=4000 result **12.075% [11.10%, 13.12%]** vs
`HeuristicAgent`, decisively below the >25% gate above, though clearly
better than v1's band, non-overlapping at the low end).

**Linked from** truco-py logs
[004](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/004-april-selfplay-collapse.md)
and
[005](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/005-june-threshold-mix-collapse.md),
each recorded as *inconclusive, confounded* by that log's own verdict: the
confirmed root cause of both truco collapses was pool contamination
([013](013-selfplay-pool-contamination.md)), not the mix ratio, so
neither run isolates the mix value as the variable under test.

Net picture: the mix ratio does appear to matter (v1 vs v2's non-overlapping
bands), but heavier baseline share alone has not yet cleared this note's own
gate for catan, and the truco evidence that originally motivated the
counter-evidence section above turned out to be measuring a different bug.
Status stays `running` pending a catan attempt that isolates mix ratio with
a clean, contamination-free pool at a higher value or longer budget (see
[009](009-longer-runs-and-resume.md)).

## Related notes

- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
- [009 — Longer runs / resume when the curve has not bent](009-longer-runs-and-resume.md)
- [011 — KL guard vs the previous snapshot](011-kl-guard.md)
- [013 — Self-play pool contamination across runs](013-selfplay-pool-contamination.md)
