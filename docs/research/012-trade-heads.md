# 012 — Enable the reserved trade heads

**Status:** idea
**Last touched:** 2026-09-20

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

Not yet attempted — the heads remain masked off as of catan PR #17/#18.

## Related notes

- [004 — Factored action head via sequential atom composition](004-factored-action-head.md)
