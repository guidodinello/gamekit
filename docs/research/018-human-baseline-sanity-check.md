# 018 — Human baseline as a sanity check for learned / hand-written agents

**Status:** planned (pre-registered in catan log 007; no games played yet)
**Last touched:** 2026-09-29

## Hypothesis

Machine-vs-machine win rates only measure strength relative to other
machines, so a small number of logged games against a human is the
cheapest way to learn whether a hand-written or learned agent is actually
good at the game, before spending on further modelling work. Per bot
kind, the question is: does one human beat 3 bots of that kind more often
than the chance rate (25% in a 4-player game)?

## Why we believe it

- Every catan number so far is machine-vs-machine: the best RL agent wins
  94.6% of 4-player games against 3 `RandomAgent`s and 20.7% against 3
  `HeuristicAgent`s at n=4000, and neither agent has been measured
  against a human — no external source; observed in catan log 007 and
  [009](009-longer-runs-and-resume.md)'s results.
- A win rate against a weak or hand-written opponent says little about
  absolute play: the RL agent's 94.6% against random play sits beside
  20.7% (below the 25% chance rate) against the heuristic — no external
  source; observed in catan
  [006](https://github.com/guidodinello/catan/blob/main/docs/experiments/006-longer-run.md).
- The gate on further RL work (for example a board-aware encoder,
  [008](008-board-aware-encoder.md)) is expensive, so a cheap external
  reference is worth having first — no external source; observed in
  catan log 007's stated motivation.

## How to test

Design as pre-registered in catan log
[007](https://github.com/guidodinello/catan/blob/main/docs/experiments/007-human-games.md)
(catan [PR #29](https://github.com/guidodinello/catan/pull/29)):

- **Arms:** one human plus 3 bots of a single kind, `rl` or `heuristic`.
  N=12 games per arm (24 games), fixed up front.
- **Pairing:** engine seeds 7001-7012, each played once per bot kind, so
  boards are paired across arms. The human is rotated over turn
  position, and the order inside a pair alternates so the human's
  learning curve is not confounded with bot kind.
- **Conditions:** domestic trades are forced to reject on the `rl` seat
  (`trade_policy: "reject_all"`, recorded per game), because its trade
  accept/reject atoms are untrained (see [012](012-trade-heads.md)). A
  game that never finishes counts as a human loss.
- **Metric:** human wins of 12 per bot kind, with a 95% Wilson interval
  (`gamekit.mc.wilson_interval`).
- **Gate (per bot kind):** lower bound above 25% means the human is
  clearly stronger; upper bound below 25% means the bot is stronger;
  otherwise not distinguishable from chance. The rl-vs-heuristic
  comparison is descriptive only.
- **Power, stated up front:** at N=12 the interval is about 45-50 points
  wide. The human needs at least 6 of 12 for the lower bound to clear 25%
  (6/12 gives [25.4%, 74.6%]); "bot stronger" needs 0 of 12 (upper bound
  24.2%). The rl-vs-heuristic contrast has essentially no power: with
  3/12 in one arm the other needs about 8/12 (8 vs 3 gives p=0.041 with
  `gamekit.mc.testing.two_proportion_test`). A null result does not show
  the bots are equally strong. These figures were recomputed with
  `gamekit.mc` when this note was written.
- **Limitations:** one human with a self-assessed level, learning over 24
  games; games are not blind; no domestic trading; fixed boards.
- **Cost estimate:** human time for 24 games; the tooling (rl bot seat,
  game records, tabulator) is already built in catan PR #29.

## Result

Not yet attempted: no games have been played and no result exists. The
pre-registration in catan log 007 is committed; its Result and Verdict
sections are empty on purpose.

**Tested by:** catan log
[007](https://github.com/guidodinello/catan/blob/main/docs/experiments/007-human-games.md)
(pending).

**Motivated by:** catan experiment 007 (catan
[PR #29](https://github.com/guidodinello/catan/pull/29), Refs catan
issue #26).

## Related notes

- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
- [008 — Board-aware encoder / spatial inductive bias](008-board-aware-encoder.md)
- [012 — Enable the reserved trade heads](012-trade-heads.md)
- [019 — Human Catan game data as a training source](019-human-catan-game-data.md)
- [020 — Modular agent: separate trade module over a strategy policy](020-modular-trade-agent.md)
