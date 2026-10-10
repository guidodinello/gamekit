# Research notes

A durable, git-tracked logbook for hypotheses about how to improve or
build better models for the games that consume `gamekit`
(`catan`, `truco-py`, and future ones). Every idea traces back to where it
came from and to the experiment that validated or rejected it. A rejected
hypothesis is as valuable a note as a validated one — `rejected` is a
first-class status, not a reason to delete a file.

## The two-layer convention

This split was established in
[gamekit#26](https://github.com/guidodinello/gamekit/pull/26) (there was no
prior convention to follow):

- **Technique notes live here, in `gamekit`.** One hypothesis per note,
  game-agnostic, backed by a literature citation or an explicit "no
  external source; observed in `<PR/run>`" — see `TEMPLATE.md`.
- **Experiment logs live in the consumer repo**, under `docs/experiments/`
  (e.g. `catan/docs/experiments/`, `truco-py/docs/experiments/`). A log
  holds the numbers: config, run id, and the stamped result JSON that
  `gamekit.results` produces. It links back to the note id here that it
  tests. See `EXPERIMENT-LOG-TEMPLATE.md` for the shape a log should take
  — copy it into the consumer repo's `docs/experiments/` directory.

A note's `Result` section links *forward* to the experiment log(s) that
tested it; an experiment log links *back* to the note id it tests. Neither
repo owns both halves of the story, so the link is what keeps them
traceable to each other.

## How to add a note

1. Copy `TEMPLATE.md` to `NNN-slug.md`, where `NNN` is the next unused
   three-digit id (see the index below) and `slug` is a short kebab-case
   title.
2. Fill in `Status` and `Hypothesis` first — those are the two fields worth
   writing even if nothing else is ready yet.
3. Add a row to the index table below, keeping it sorted by id.
4. Update `Last touched` whenever you edit the note, even just to change
   its status.

`TEMPLATE.md`'s sections are required; add extra sections where a note
genuinely needs them (several seed notes below add one, e.g. a
"Counter-evidence" or "Why this is still `idea`" section) — the fixed
sections are a floor, not a ceiling.

## Status vocabulary

| Status | Meaning |
|---|---|
| `idea` | Not yet planned — a hypothesis worth writing down. |
| `planned` | The test is designed; no run has started. |
| `running` | An experiment is in progress or has interim results. |
| `tested` | The pre-registered run completed, but the verdict was neither validated nor rejected. Add a qualifier in parentheses, e.g. `tested (inconclusive)`. |
| `validated` | The experiment confirmed the hypothesis. |
| `rejected` | The experiment contradicted the hypothesis. |

## Index

| id | title | status | last touched |
|---|---|---|---|
| [001](001-self-play-opponent-mix.md) | Self-play opponent mix vs a fixed baseline | tested (truco 009: mix lost held out, one seed) | 2026-10-05 |
| [002](002-bc-warm-start.md) | Behavior-cloning warm start before PPO | running | 2026-10-05 |
| [003](003-discount-horizon.md) | Discount horizon vs episode length | validated | 2026-09-20 |
| [004](004-factored-action-head.md) | Factored action head via sequential atom composition | validated | 2026-09-21 |
| [005](005-eval-statistics.md) | Eval statistics: Wilson intervals and eval-in-loop | validated | 2026-09-30 |
| [006](006-uniform-atoms-baseline.md) | Uniform-over-atoms baseline is not `RandomAgent` | validated | 2026-09-20 |
| [007](007-inference-thread-oversubscription.md) | Inference-worker thread oversubscription | validated | 2026-09-21 |
| [008](008-board-aware-encoder.md) | Board-aware encoder / spatial inductive bias | idea | 2026-09-23 |
| [009](009-longer-runs-and-resume.md) | Longer runs / resume when the curve has not bent | tested (inconclusive) | 2026-10-05 |
| [010](010-entropy-schedule.md) | Entropy schedule instead of a fixed coefficient | tested (010b rejected, one seed) | 2026-09-30 |
| [011](011-kl-guard.md) | KL guard vs the previous snapshot | running | 2026-10-05 |
| [012](012-trade-heads.md) | Enable the reserved trade heads | idea | 2026-09-29 |
| [013](013-selfplay-pool-contamination.md) | Self-play pool contamination across runs | validated | 2026-10-05 |
| [014](014-per-hand-vs-match-win-rate.md) | Per-hand win-rate ceiling vs match-play compounding | idea | 2026-09-27 |
| [015](015-shaped-reward-breaks-rew-proxy.md) | Reward shaping invalidates the `ep_rew_mean` win-rate proxy | validated | 2026-09-21 |
| [016](016-positional-advantage-rotation.md) | Positional (mano) advantage must be rotated out of a benchmark arm | validated (mechanism) | 2026-09-21 |
| [017](017-amdahl-ceiling-and-real-batch-size.md) | Measure the Amdahl ceiling and the real batch size before building an inference server | validated | 2026-09-23 |
| [018](018-human-baseline-sanity-check.md) | Human baseline as a sanity check for learned / hand-written agents | planned | 2026-09-29 |
| [019](019-human-catan-game-data.md) | Human Catan game data as a training source | idea | 2026-09-29 |
| [020](020-modular-trade-agent.md) | Modular agent: separate trade module over a strategy policy | idea | 2026-09-30 |
| [021](021-decision-time-search.md) | Decision-time search for Catan: ISMCTS with the trained policy/value network as priors | tested (upper bound; self-model confirmation pending) | 2026-10-02 |
| [022](022-league-ratings.md) | League ratings: anchored Bradley-Terry/Elo over a round-robin, with the win matrix alongside | tested (descriptive; truco 010 and catan 013, no verdict) | 2026-10-10 |
| [023](023-league-selfplay.md) | League self-play: PFSP opponent sampling from snapshots and fixed baselines | planned | 2026-10-07 |
