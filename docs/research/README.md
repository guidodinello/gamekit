# Research notes

A durable, git-tracked logbook for hypotheses about how to improve or
build better models for the games that consume `gamekit`
(`catan`, `truco-py`, and future ones). Every idea traces back to where it
came from and to the experiment that validated or rejected it. A rejected
hypothesis is as valuable a note as a validated one — `rejected` is a
first-class status, not a reason to delete a file.

## The two-layer convention

This PR establishes the split (there was no prior convention to follow):

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
| `validated` | The experiment confirmed the hypothesis. |
| `rejected` | The experiment contradicted the hypothesis. |

## Index

| id | title | status | last touched |
|---|---|---|---|
| [001](001-self-play-opponent-mix.md) | Self-play opponent mix vs a fixed baseline | running | 2026-09-20 |
| [002](002-bc-warm-start.md) | Behavior-cloning warm start before PPO | planned | 2026-09-20 |
| [003](003-discount-horizon.md) | Discount horizon vs episode length | validated | 2026-09-20 |
| [004](004-factored-action-head.md) | Factored action head via sequential atom composition | validated | 2026-09-20 |
| [005](005-eval-statistics.md) | Eval statistics: Wilson intervals and eval-in-loop | validated | 2026-09-20 |
| [006](006-uniform-atoms-baseline.md) | Uniform-over-atoms baseline is not `RandomAgent` | validated | 2026-09-20 |
| [007](007-inference-thread-oversubscription.md) | Inference-worker thread oversubscription | validated | 2026-09-20 |
| [008](008-board-aware-encoder.md) | Board-aware encoder / spatial inductive bias | idea | 2026-09-20 |
| [009](009-longer-runs-and-resume.md) | Longer runs / resume when the curve has not bent | idea | 2026-09-20 |
| [010](010-entropy-schedule.md) | Entropy schedule instead of a fixed coefficient | idea | 2026-09-20 |
| [011](011-kl-guard.md) | KL guard vs the previous snapshot | idea | 2026-09-20 |
| [012](012-trade-heads.md) | Enable the reserved trade heads | idea | 2026-09-20 |
| [013](013-selfplay-pool-contamination.md) | Self-play pool contamination across runs | validated | 2026-09-21 |
| [014](014-per-hand-vs-match-win-rate.md) | Per-hand win-rate ceiling vs match-play compounding | idea | 2026-09-21 |
| [015](015-shaped-reward-breaks-rew-proxy.md) | Reward shaping invalidates the `ep_rew_mean` win-rate proxy | validated | 2026-09-21 |
| [016](016-positional-advantage-rotation.md) | Positional (mano) advantage must be rotated out of a benchmark arm | validated (mechanism) | 2026-09-21 |
