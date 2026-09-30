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

## End-to-end joint training (intended long-term path)

Guido's stated long-term intent (2026-09-29): unmask the trade heads and
train strategy and trading jointly, as one policy, against opponents that
trade, so the two co-adapt. Guido calls this the purest path. It is what
this note's hypothesis already describes; this section records that it is
the direction we intend to implement, not just a candidate.

- **Rationale:** co-adaptation (what a trade is worth depends on the
  strategic plan, and the plan depends on which trades are available) and
  no hand-designed module boundary to get wrong — no external source;
  reasoning about the game, untested.
- **Risks:**
  - *Sparse trade reward:* trades are rare next to other decisions and
    pay off only through the eventual win.
  - *Non-stationary trading opponents:* if opponents are also learning
    (self-play with unmasked heads), the counterparty distribution keeps
    moving.
  - *Harder credit assignment:* a trade's effect shows up many turns later.
  - *Larger action space:* the give/receive heads enlarge the head that
    has to be learned within a comparable budget.
- **Shared prerequisite:** opponents that actually propose trades, plus a
  trade-aware benchmark (catan
  [#28](https://github.com/guidodinello/catan/issues/28)). Without them
  the accept/reject head gets no gradient and the propose/counter heads
  have nobody to trade with.
- **Relation to [020](020-modular-trade-agent.md):** the modular stages
  there are stepping stones and baselines for this path, not a
  replacement. For example, 020 stage 1 (value-priced responses with no
  trade training) is the baseline end-to-end training has to beat.
  Human data ([019](019-human-catan-game-data.md)) could supply
  trading-opponent behavior or an acceptance model for either path.

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

Not yet attempted — the propose/counter heads remain masked off, and no
trading opponent exists in training (one now exists for benchmarking, see
below). Tracked by catan
[#28](https://github.com/guidodinello/catan/issues/28).

**Tested by / Linked from:** catan log
[008](https://github.com/guidodinello/catan/blob/main/docs/experiments/008-trading-opponents-baseline.md)
(catan [PR #36](https://github.com/guidodinello/catan/pull/36)) —
prerequisite only: trading opponents + trade-aware benchmark + baseline;
no trade learning. Numbers as in the log (n=4000 per arm, seat-rotated,
Wilson 95%):

- 2 `TradingHeuristic` + 2 `Heuristic`: the trading seats won 53.4%
  [51.85%, 54.94%] of games — trading helps a heuristic (small effect).
- 1 `Heuristic` vs 3 `TradingHeuristic`: the heuristic won 21.5%
  [20.28%, 22.83%].
- **Baseline for every trade stage:** `rl` (trade responses forced to
  reject) vs 3 `TradingHeuristic`: **18.9% [17.72%, 20.14%]**, 1.73
  completed trades per game. The end-to-end path has to beat this with a
  non-overlapping interval.
- Caveat: the opponent's acceptance rate is only 4-7% of proposals; it is
  a minimal rule, not a model of human trading.

**Finding — the RL checkpoint always plays `PlayVictoryPoint`.** In log 008
arm C the RL seat plays it at every legal opportunity (3854 of 3854
legal decisions), cloned from `HeuristicAgent` (catan
[#34](https://github.com/guidodinello/catan/issues/34)). Relevant to the
next action-space change, since that atom is a candidate to mask
alongside enabling the trade heads.

**Linked from** catan log
[007](https://github.com/guidodinello/catan/blob/main/docs/experiments/007-human-games.md),
which does not test this note: it forces reject on the `rl` seat and
names a trade-learning agent as the condition for a fair human match with
trading.

## Related notes

- [004 — Factored action head via sequential atom composition](004-factored-action-head.md)
- [018 — Human baseline as a sanity check for learned / hand-written agents](018-human-baseline-sanity-check.md)
- [019 — Human Catan game data as a training source](019-human-catan-game-data.md)
- [020 — Modular agent: separate trade module over a strategy policy](020-modular-trade-agent.md)
- [021 — Decision-time search for Catan: ISMCTS with the trained policy/value network as priors](021-decision-time-search.md)
