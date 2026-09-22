# 008 — Board-aware encoder / spatial inductive bias

**Status:** idea
**Last touched:** 2026-09-21

## Hypothesis

An encoder that respects the board's graph structure (which vertices
touch which edges and hexes) should generalize better and/or train faster
than a flat vector of per-entity feature blocks that discards adjacency.

## Why we believe it

- Catan's board is explicitly graph-structured — "a hexagonal board where
  each vertex, edge and face has its own features" — [Gendre & Kaneko
  2020, *Playing Catan with Cross-dimensional Neural
  Network*](https://arxiv.org/abs/2008.07079) — takeaway: the literature
  treats this structure as central enough to motivate a custom
  ("cross-dimensional") network architecture, not an incidental detail a
  flat encoder can safely ignore.
- Structured representations that operate over relations (nodes, edges)
  rather than flat feature vectors are the general case for this kind of
  inductive bias — [Battaglia et al. 2018, *Relational inductive biases,
  deep learning, and graph
  networks*](https://arxiv.org/abs/1806.01261) — takeaway: a graph-network
  encoder is a well-studied, named alternative to a flat vector, not a
  from-scratch invention.
- catan's current encoder is a flat concatenation of fixed-width blocks —
  hex features, vertex features, edge features, each block just a
  repetition of per-entity feature vectors with no explicit adjacency
  structure fed to the network (19 hexes, 54 vertices, 72 edges,
  1484-dim total) — no external source; observed in catan
  `rl/encoder.py:68-108`.

## How to test

- **Metric:** win rate and/or sample efficiency (win rate reached at a
  fixed step budget) vs the current flat encoder, same training budget,
  same opponent.
- **Gate:** a graph-aware encoder should reach the same win-rate gate in
  fewer steps, or a materially higher win rate at the same step budget,
  with non-overlapping CIs.
- **Cost estimate:** high — a new encoder architecture plus at least one
  full training run to compare against the existing flat-encoder baseline;
  this is why it's still an idea rather than planned.

## Result

Not yet attempted.

**Motivated by:** catan log
[003](https://github.com/guidodinello/catan/blob/main/docs/experiments/003-selfplay-v2-baseline-mix-0.5.md)
names this as an untried direction its own win-rate ceiling motivates
(12.075% vs `HeuristicAgent`, gate not met) — the log does not test this
note's hypothesis, it only points at it as a candidate next step.

## Related notes

- [004 — Factored action head via sequential atom composition](004-factored-action-head.md)
