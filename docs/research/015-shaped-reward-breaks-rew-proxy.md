# 015 — Reward shaping invalidates the `ep_rew_mean` win-rate proxy

**Status:** validated
**Last touched:** 2026-09-21

## Hypothesis

Treating `ep_rew_mean` as a stand-in for win rate via a formula like
`(1 + r) / 2` silently breaks the moment a per-step reward-shaping term is
added, because the formula assumes an episode return bounded in
`[-1, 1]` (one win/loss/draw signal per episode) and shaping accumulates a
term every step instead.

## Why we believe it

- SB3 defines `ep_rew_mean` as "Mean episodic training reward (averaged
  over `stats_window_size` episodes, 100 by default)" — the raw,
  undiscounted sum of whatever reward the environment returns each step,
  with no assumption about its scale — [Stable-Baselines3, Logger
  docs](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)
  — takeaway: the metric itself makes no promise about being bounded; any
  bounded-return assumption has to come from the reward function, not
  from SB3.
- Potential-based shaping is provably policy-invariant — it changes *how
  fast* the signal arrives, not which policy is optimal — [Ng, Harada &
  Russell 1999, *Policy Invariance Under Reward Transformations: Theory
  and Application to Reward
  Shaping*](https://dl.acm.org/doi/10.5555/645528.657613) — takeaway: the
  invariance guarantee is over the *optimal policy*, not over the
  *return's scale*. A shaping term is free to push the accumulated return
  arbitrarily far outside whatever range a downstream proxy formula
  assumed, without that being evidence of anything going wrong in
  training.
- truco-py documents exactly the formula this breaks: "`ep_rew_mean` |
  Win rate proxy: `(1 + valor) / 2`" — no external source; observed in
  truco-py `README.md:196`.
- truco-py's `ShapedReward.compute` returns `sparse + weight *
  (delta_mine - delta_opp)` **per step**, so a 165-step episode
  accumulates a return far outside `[-1, 1]` — no external source;
  observed in truco-py `training/reward.py:65-83`.
- The June self-play collapse produced `ep_rew_mean` of -7.28 and -6.18
  under `--shaped-reward` — values the documented proxy formula cannot
  map to a win-rate percentage at all, let alone a sane one — no external
  source; observed in truco-py logs
  [004](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/004-april-selfplay-collapse.md)
  and
  [005](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/005-june-threshold-mix-collapse.md).
  This is a monitoring pitfall adjacent to
  [005](005-eval-statistics.md) and [011](011-kl-guard.md): the training
  itself is not necessarily broken (shaping is policy-invariant by
  construction), but reading `ep_rew_mean` as a live win-rate signal is,
  the moment shaping is on.

## How to test

- **Metric:** whether the reward function returns a per-episode value
  (bounded, one win/loss/draw signal) or a per-step accumulation (shaping,
  unbounded across episode length).
- **Gate:** any `ep_rew_mean`-derived win-rate proxy is only valid when
  the reward is sparse/per-episode; a shaped reward needs an actual
  win-rate eval callback (see [005](005-eval-statistics.md)), not a
  formula applied to the training reward.
- **Cost estimate:** near zero — this is a documentation/interpretation
  fix, not a training change; the eval-in-loop machinery
  [005](005-eval-statistics.md) already argues for supplies the correct
  signal.

## Result

**Tested by:** truco-py log
[005](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/005-june-threshold-mix-collapse.md)
shows the break directly (`ep_rew_mean` -7.28/-6.18 under
`--shaped-reward`, against `README.md:196`'s stale proxy formula); log
[004](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/004-april-selfplay-collapse.md)
notes the same formula is stale for that run too.

## Related notes

- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
- [011 — KL guard vs the previous snapshot](011-kl-guard.md)
