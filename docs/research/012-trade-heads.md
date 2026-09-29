# 012 — Enable the reserved trade heads

**Status:** idea
**Last touched:** 2026-09-29

## Hypothesis

Enabling the currently-masked-off `ProposeTrade`/`CounterTrade` atom
heads should improve a trained agent's win rate against opponents that
themselves trade, since the agent currently gives up the initiative half
of a mechanic its opponents use freely.

## Why we believe it

- The action-space design deliberately reserved these heads rather than
  omitting them, specifically so enabling them later would be "a mask flip
  rather than a reshaping of the policy head" — no external source;
  observed in catan `rl/action_space.py` (PR #17).
- The known bias this creates is already measured and documented: "The RL
  agent never *proposes* a domestic trade... It does still accept and
  reject offers, so it is not excluded from trading entirely — but against
  3 RandomAgents, which do propose, it gives up the initiative half of a
  rule random play uses constantly." — no external source; observed in
  catan `experiments/results/benchmark_rl_vs_random_p4.json`'s
  `known_biases[3]` (PR #18).

## Refinement (2026-09-29)

Found while planning catan experiment 007 (catan
[PR #29](https://github.com/guidodinello/catan/pull/29)) and tracked in
catan issue [#28](https://github.com/guidodinello/catan/issues/28):

- **The accept/reject atoms are live but untrained.** Only the
  propose/counter sentinels are masked. The RL agent still has live
  accept and reject atoms, but no training opponent ever proposed a trade
  to it: behavior cloning copied a `HeuristicAgent` that never proposes
  and always rejects, and the self-play opponents (heuristic and RL) never
  propose either. Those atoms have therefore received no gradient in this
  checkpoint's lineage. The issue text that called them "masked" was
  inaccurate — no external source; observed in catan PR #29.
- **Trading opponents are a prerequisite, not just unmasking.** Flipping
  the mask alone would leave the accept/reject head untrained, and the
  propose/counter heads would have nobody to trade with. Training needs at
  least one opponent that proposes: a trading heuristic variant, RL
  self-play with the heads unmasked, or both (catan #28).
- **Bearing on the gate below.** The vs-`RandomAgent` benchmarks did
  exercise the untrained accept head, because `RandomAgent`s propose;
  the vs-`HeuristicAgent` benchmark did not. So the two benchmarks differ
  in more than opponent strength, and catan #28 also asks for a
  trade-aware benchmark plus trade frequency and acceptance stats.
- **Interim handling.** Catan log
  [007](https://github.com/guidodinello/catan/blob/main/docs/experiments/007-human-games.md)
  forces the `rl` seat's trade responses to reject
  (`trade_policy: "reject_all"`), which matches the conditions of the
  n=4000 rl-vs-heuristic benchmark. This is temporary; see
  [018](018-human-baseline-sanity-check.md).

## How to test

- **Metric:** win rate vs opponents that propose trades (at minimum
  `RandomAgent`, ideally `HeuristicAgent` once [001](001-self-play-opponent-mix.md)
  has a committed gate run), with and without the trade heads enabled.
- **Gate:** a measurable, non-overlapping-CI win-rate improvement from
  enabling the heads, at the same step budget.
- **Cost estimate:** the mask flip itself is cheap (by design); the cost
  is a full training run to see whether the larger, previously-unused
  give/receive action space actually gets learned productively within a
  comparable budget, or instead slows convergence.

## Result

Not yet attempted — the propose/counter heads remain masked off as of
catan PR #29, and no trading opponent exists in training. Tracked by
catan [#28](https://github.com/guidodinello/catan/issues/28).

**Linked from** catan log
[007](https://github.com/guidodinello/catan/blob/main/docs/experiments/007-human-games.md),
which does not test this note: it forces reject on the `rl` seat and
names a trade-learning agent as the condition for a fair human match with
trading.

## Related notes

- [004 — Factored action head via sequential atom composition](004-factored-action-head.md)
- [018 — Human baseline as a sanity check for learned / hand-written agents](018-human-baseline-sanity-check.md)
