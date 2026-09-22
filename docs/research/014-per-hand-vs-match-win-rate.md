# 014 — Per-hand win-rate ceiling vs match-play compounding

**Status:** idea
**Last touched:** 2026-09-21

## Hypothesis

In a high-variance card game, per-hand win rate against a fixed opponent
has a ceiling well below what the same skill gap reaches over a full
match, because a single hand is dominated by deal variance while skill
compounds across the many hands a match actually plays out over. A
win-rate gate that doesn't name which unit it's measuring (per-hand vs.
per-match) can look unreachable at one unit and comfortably clearable at
the other.

## Why we believe it

- truco-py observed this directly: "Single-hand Truco is dominated by
  deal variance. Even the best agent vs the worst opponent caps at ~70%
  per hand. 90%+ is only achievable in full match play, where skill
  compounds across hands." — no external source; observed in truco-py
  `docs/session-2026-06-05.md:27-30`.
- The same session's own numbers show the gap concretely: VonNeumann(r=20)
  reached ~65-70% per hand vs Random and vs Threshold, but 67% and 85.3%
  respectively once measured over a full match to 40 points — no external
  source; observed in truco-py `docs/session-2026-06-05.md:44-58`.
- This matches an independently-derived result from real-money poker: the
  point at which the top performers pull statistically ahead of the
  bottom performers takes on the order of 1,500 hands, not a handful —
  [Potter van Loon, van den Assem & van Dolder 2015, *Beyond Chance? The
  Persistence of Performance in Online
  Poker*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0115479),
  PLOS ONE 10(3) — takeaway: "skill predominates after approximately 1,500
  hands" is the same phenomenon truco-py observed at a smaller scale (one
  match vs. one hand) — variance dominates a short sample regardless of
  the underlying skill gap, and the sample has to grow before skill shows
  through.

## No experiment log covers this yet

**No truco-py experiment log tests this hypothesis directly.** The ~70%
figure lives only in a session note
(`docs/session-2026-06-05.md:27-30,44-48`), not in any of the seven
fact-checked logs in
[`docs/experiments/`](https://github.com/guidodinello/truco-py/tree/main/docs/experiments) —
see that folder's README, caveat 4: `scripts/benchmark.py` never persisted
its output to a stamped result, so every number in the source material is
hand-copied prose. The per-hand baselines table itself
(`docs/session-2026-06-05.md:44-48`) records `n` only as a range
(500-1000) applied to the whole table, never per row, so no confidence
interval can be computed for the ~70% figure specifically. That is the
reason this stays `idea` rather than `planned`: the hypothesis is
plausible and doubly evidenced (in-repo and in the literature), but no
run in either consumer repo has been designed to measure it with a
reported `n` and interval.

## How to test

- **Metric:** win rate vs a fixed opponent, measured two ways in the same
  run — per-hand and per-match — with Wilson intervals for both (see
  [005](005-eval-statistics.md)).
- **Gate:** if a win-rate gate is meant to certify match-level competence,
  it should be defined and measured at the match level, not the hand
  level; a per-hand number should be reported alongside as context, not
  substituted for it.
- **Cost estimate:** near zero beyond an existing benchmark run — the
  same games already played can be scored at both units if the harness
  tracks match boundaries, which `gamekit.benchmark` already does via its
  `n_games` unit.

## Result

Not yet attempted with a reported `n` and interval in either consumer
repo.

## Related notes

- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
