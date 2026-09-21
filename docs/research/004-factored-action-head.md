# 004 — Factored action head via sequential atom composition

**Status:** validated
**Last touched:** 2026-09-20

## Hypothesis

For a game whose legal-action set is neither small nor fixed-shape (a
multi-parameter action like "propose this trade" or "build this road
pair"), composing an action as a short sequence of masked picks from one
flat `Discrete` head — re-deriving the mask from the real engine at each
sub-step — handles conditional legality that `MultiDiscrete` cannot.

## Why we believe it

- sb3-contrib's `MultiDiscrete` masking splits one flat mask columnwise
  and applies each sub-head's mask independently, with **no cross-head
  constraint mechanism**: `MaskableMultiCategoricalDistribution
  .apply_masking` reshapes the mask and does
  `th.split(masks_tensor, action_dims, dim=1)`, then applies each split to
  its own sub-distribution — [sb3-contrib, *Maskable
  PPO*](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
  — takeaway: every sub-head's mask must be computable *before* any
  sub-head is sampled, so a second choice that depends on the first (e.g.
  a road-building pair where placing the first edge changes which second
  edges are legal) cannot be expressed as one `MultiDiscrete` step.
- The general formulation of decomposing a large action space into
  sequential sub-decisions, each with its own mask, is Conditional Action
  Trees — [Bamford & Ovalle 2021, *Generalising Discrete Action Spaces with
  Conditional Action Trees*](https://arxiv.org/abs/2104.07294) — takeaway:
  "decomposing it into multiple sub-spaces, favoring a multi-staged
  decision making approach" is a named, studied pattern, not an ad hoc
  workaround.
- Catan's action space is characterized in the literature as unusually
  wide and heterogeneous — "a large action space (including negotiation)"
  over "a hexagonal board where each vertex, edge and face has its own
  features" — [Gendre & Kaneko 2020, *Playing Catan with Cross-dimensional
  Neural Network*](https://arxiv.org/abs/2008.07079) — takeaway: this is a
  domain where a flat, pre-enumerated action list is a real design
  tradeoff, not a strawman.
- catanatron, a widely used Catan RL environment, takes the alternative
  path of a flat enumerated action list with masking via
  `info["valid_actions"]` — [catanatron](https://github.com/bcollazo/catanatron)
  — takeaway: flat enumeration is workable when the action's parameters
  are small and finite, which is exactly the case sequential atoms exist
  to handle when they are not (catan's `Discard` reaches 197 legal
  multisets in a single decision, and `_road_building_pairs` emits
  *ordered* pairs 106 wide).
- catan measured that `_road_building_pairs` recomputes legality with the
  first edge already placed, so the second edge's mask genuinely cannot be
  known before the first pick — no external source; observed in catan
  `rl/action_space.py:15-19` (PR #17).

## How to test

- **Metric:** exact round-trip (every legal action encodes to an atom
  sequence and back) and DFS-reachability (an unmasked-atom search reaches
  exactly the legal action set — nothing lost, nothing invented) over a
  batch of full random games.
- **Gate:** both properties hold with zero exceptions across the swept
  games; masks are derived from the real engine's `legal_actions()`, never
  from a reimplementation of the rules.
- **Cost estimate:** a static/structural test, not a training run — cheap
  relative to any RL experiment that depends on it being correct.

## Result

catan PR #17: one flat 223-wide `Discrete` head over atoms, with the
round-trip and DFS-reachability properties swept over 12 full random games
across all 9 phases, plus 237 passing tests (194 existing + 43 new). Trade
heads (`ProposeTrade`/`CounterTrade`) are reserved but masked off pending
[012](012-trade-heads.md).

## Related notes

- [006 — Uniform-over-atoms baseline is not `RandomAgent`](006-uniform-atoms-baseline.md)
- [012 — Enable the reserved trade heads](012-trade-heads.md)
