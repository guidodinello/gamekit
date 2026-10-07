# 023 — League self-play: PFSP opponent sampling from snapshots and fixed baselines

**Status:** planned
**Last touched:** 2026-10-07

## Hypothesis

Sampling training opponents from a league (the learner's own past snapshots plus fixed baseline agents), weighted by each member's win rate against the learner and with the baselines always kept in the mix, gives better play against held-out opponents than baseline-only training or a fixed self/baseline split.

## Why we believe it

- Naive self-play can "chase cycles (for example, where A defeats B, and B defeats C, but A loses to C) indefinitely without making progress"; AlphaStar's main agents use "a prioritized fictitious self-play (PFSP) mechanism that adapts the mixture probabilities proportionally to the win rate of each opponent against the agent" — [Vinyals et al. 2019, *Grandmaster level in StarCraft II using multi-agent reinforcement learning*](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf) (quotes as verified in [022](022-league-ratings.md), which says the Nature page itself was not read) — takeaway: the win-rate matrix that `gamekit.league` produces is the input a PFSP sampler needs. Only these two points are taken from the paper; its exact weighting function is not used here.
- A fixed, nonzero share of older policies is a standard simpler form of the same idea — [*Dota 2 with Large Scale Deep Reinforcement Learning*, arXiv:1912.06680](https://arxiv.org/abs/1912.06680) (80% latest policy / 20% older policies; as quoted in [001](001-self-play-opponent-mix.md) and [022](022-league-ratings.md)) — takeaway: the open question is how to choose the mixture, not whether to mix.
- Self-play improved play against similar agents but, in truco, did not carry over to a different-style opponent. truco-py's arm **M** (opponent team per episode Threshold 0.4 / Random 0.2 / own snapshot 0.4; partners = latest own snapshot) lost to the baseline-only arm **C** against the held-out VonNeumann agent, 56.8% [55.3, 58.3] vs 64.1% [62.6, 65.5] (n=4000 each, seat-rotated, one seed per arm, M − C = −7.28 pts, 95% CI [−9.41, −5.14]), while the two tie against Threshold (91.8% vs 91.2%) — no external source; observed in truco-py [log 009](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/009-retrain-mixed-pool.md).
- In the same experiment M changed two things at once (self share 0.4 **and** Random share 0.2), and partners were own snapshots in **both** arms, so the cause of the gap is **not** isolated; [001](001-self-play-opponent-mix.md) already records this and proposes the ablation that is gate 1 below — no external source; observed in truco-py log 009.
- Head to head, M beats C at every matched step (M20 vs C20 58.9%, i.e. C20 wins 41.1% [40.1, 42.0]) and M20 rates highest on the league scale (Elo 353 vs C20 308, Threshold anchored at 0; no significant 3-cycles; VonNeumann was **not** in that roster) — no external source; observed in truco-py [log 010](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/010-league-009-checkpoints.md). Read together with the 009 numbers: measured against similar agents M is better, measured against a different-style opponent it is worse, which is the single-yardstick problem a league is meant to expose.
- Neither 009 arm beat its teacher on the held-out opponent: Threshold itself beats VonNeumann 64.6% [63.1, 66.1] against C's 64.1% — no external source; observed in truco-py log 009 (post-hoc, not pre-registered).
- Team games add a second moving part. When partner seats are filled from the learner's own snapshots, the partner is learned as well, so the training distribution can shift through the partner as well as through the opponent. 009 used snapshot partners in both arms and did not test this — no external source; hypothesis only, observed nowhere yet.
- Earlier truco self-play collapses (April run at mix 0.2: 56.0% (n=150) / 57.3% (n not recorded) vs Threshold; June run at mix 0.5, frozen in training metrics) are **caveated history, not evidence for or against this note**: those numbers predate seat rotation and the truco rules fixes, and the confirmed cause was cross-run pool contamination ([013](013-selfplay-pool-contamination.md)), not self-play itself. Seat-rotated, the April checkpoint scores 2154/4000 (53.9%) and the June checkpoints 70.0–78.5% vs Threshold — no external source; observed in truco-py logs [004](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/004-april-selfplay-collapse.md), [005](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/005-june-threshold-mix-collapse.md) and [008](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/008-seat-rotated-rebenchmarks.md).

## How to test

- **Metric:** win rate against held-out opponents (agents never in any training mix), Wilson 95% CI ([005](005-eval-statistics.md)); plus league rating, win matrix and cycle report over a roster that includes the held-out agents ([022](022-league-ratings.md)).
- **Comparison:** the league arm against a baseline-only arm (009's C) under the same budget, with more than one seed per arm. Held-out judging is always on agents the learner never trained against. Baselines stay in the sampling mix in the league arm.
- **Gate (do these first, in order):**
  1. The ablation proposed in [001](001-self-play-opponent-mix.md), each against C with more than one seed: `--opponent-mix thr=0.8,rand=0.2,self=0` (isolates the self-opponent share) and `--partners threshold` (isolates the self-partner half). If neither self opponents nor self partners hurt, the motivation for a league weakens.
  2. truco-py experiment 011, a league that adds VonNeumann to the roster ([pre-registered, not yet run](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/011-league-vonneumann.md)).
- **Cost estimate:** a bigger project than a flag change. It likely needs gamekit-side support: an opponent sampler driven by `gamekit.league` ratings and win matrix, which is what [gamekit#39](https://github.com/guidodinello/gamekit/issues/39) already tracks (builds on run-scoped pools, [013](013-selfplay-pool-contamination.md)). Named here, not designed. Training cost is that of the 009 arms (log 009 estimated about 6.6 h of laptop wall time for its two concurrent 20M-step arms) times the number of seeds, plus league evaluation.

## Result

Not yet run. This note is gated on the two items above.

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
- [013 — Self-play pool contamination across runs](013-selfplay-pool-contamination.md)
- [022 — League ratings: anchored Bradley-Terry/Elo over a round-robin, with the win matrix alongside](022-league-ratings.md)
