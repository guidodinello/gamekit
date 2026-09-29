# 020 — Modular agent: separate trade module over a strategy policy

**Status:** idea
**Last touched:** 2026-09-29

## Hypothesis

A rule-routed modular agent — trade decisions go to a trade module,
everything else to the strategy policy — can learn to trade more cheaply
and more safely than training the trade heads end to end. This is a
hierarchical split by decision type, not a mixture of experts with a
learned gate.

## Why we believe it

- A degenerate form already exists: the `rl` seat in catan experiment 007
  routes trade *responses* to `HeuristicAgent` (always reject) and uses the
  policy for everything else, because the policy's accept/reject atoms
  were never trained — no external source; observed in catan log
  [007](https://github.com/guidodinello/catan/blob/main/docs/experiments/007-human-games.md)
  (catan [PR #29](https://github.com/guidodinello/catan/pull/29)) and
  [012](012-trade-heads.md)'s Refinement section.
- The modular split lets each part be benchmarked against the previous one
  and gives a fallback if learning to trade fails — no external source;
  this is a design argument, not an observation.
- **Key risk:** the value of a trade depends on the strategic plan (which
  resources the agent needs for its next build), so a trade module trained
  independently lacks that context and may accept or propose trades that
  fit no plan — no external source; reasoning about the game, untested.

## Staged path

Each stage is benchmarked against the previous one, against opponents that
actually trade.

1. **Value-priced responses.** Accept an offer iff V(s after trade) >
   V(s) + margin, using the strategy model's critic. No trade training.
   Cheapest first experiment; compare against forced reject.
2. **Proposals by enumeration.** Enumerate candidate trades, rank them by
   the same V, and propose the best.
3. **Opponent-acceptance model.** Predict which offers a human-like
   opponent accepts, to pick offers likely to be taken (data from
   [019](019-human-catan-game-data.md), e.g. STAC).
4. **Trained trade module, strategy frozen.**
5. **Optional joint fine-tuning.** The alternative is the end-to-end path
   of unmasking the trade heads and training everything together
   ([012](012-trade-heads.md), section "End-to-end joint training").

**Caveat for stage 1:** V comes from a critic trained on games without
trades, so it may be miscalibrated on post-trade states it has never seen.
Check calibration before trusting the margin.

**Relation to the end-to-end path:** the stages are stepping stones and
baselines for 012's end-to-end joint training, not a replacement for it.
Stage 1 in particular is the baseline end-to-end training has to beat.

## How to test

- **Prerequisite shared with #28:** a benchmark against opponents that
  actually trade (a trading heuristic variant and/or self-play with the
  heads unmasked), reporting trade frequency and acceptance stats. Without
  it, no stage can be measured.
- **Metric:** win rate with Wilson intervals
  ([005](005-eval-statistics.md)) in 4-player games against trading
  opponents; per stage, versus the previous stage. Stage 1 versus forced
  reject (`trade_policy: "reject_all"`).
- **Gate:** each stage must beat the previous one with non-overlapping
  intervals to justify the added complexity.
- **Cost estimate:** stage 1 is inference-only with the existing
  checkpoint plus a trading opponent; later stages need training runs.

## Result

Not yet attempted. Tracked by catan issue
[#28](https://github.com/guidodinello/catan/issues/28).

**Motivated by:** Guido's hypothesis while planning catan #28, and the
degenerate routing in catan log 007.

## Related notes

- [012 — Enable the reserved trade heads](012-trade-heads.md)
- [018 — Human baseline as a sanity check for learned / hand-written agents](018-human-baseline-sanity-check.md)
- [019 — Human Catan game data as a training source](019-human-catan-game-data.md)
