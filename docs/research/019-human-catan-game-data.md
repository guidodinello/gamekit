# 019 — Human Catan game data as a training source

**Status:** idea
**Last touched:** 2026-09-29

## Hypothesis

Logged human Catan games are too few to fix the spatial-placement
bottleneck of behavior cloning, but they are the most relevant data we
could get for *trading* (catan
[#28](https://github.com/guidodinello/catan/issues/28)): humans trade
constantly and no bot of ours does. The most promising use is to extract
priors (action-type preferences, an opponent-acceptance model) rather than
to clone a human policy.

## Why we believe it

- The largest public source, colonist.io, is off-limits for training. Its
  terms forbid "Scraping, crawling, or indexing any portion of the Service,
  including but not limited to user profiles, game replays, leaderboards, or
  game statistics", forbid using internal APIs "unless explicitly authorized
  in writing", and forbid "Using any Content or data from the Service to
  train, develop, or improve any artificial intelligence (AI) models,
  machine learning algorithms, or automated decision-making systems without
  the Company's express prior written authorization"; the terms mention no
  official API or data export (last updated 2026-04-30, retrieved
  2026-09-29) — [Colonist.io Terms of Use](https://colonist.io/terms) —
  takeaway: colonist data needs written permission before any training use.
- A fan-made extension can export a user's own colonist games as JSON, but
  it is not official and does not lift the clause above — [ColonyHistorian
  README](https://github.com/lemeryfertitta/ColonyHistorian) (saves
  `historian-<game-id>.json` per finished game; web viewer at
  <https://lemeryfertitta.github.io/ColonyHistorian/>; Chrome listing:
  [Colony Historian](https://www.chromeboard.com/extension/colony-historian-olhlbckekdlekoaifaoboooalbeboefj))
  — takeaway: a technical export path exists, a legal one does not.
- The STAC corpus is 45 human games (modified JSettlers) with annotated
  trade-negotiation chat, under CC BY-NC-SA 4.0 — [STAC corpus
  page](https://www.irit.fr/STAC/corpus.html) — takeaway: usable for
  non-commercial research with attribution and share-alike. The "situated"
  tables (`situated_only_tables.zip`) include server/UI event rows as
  text — builds, dice rolls, "made an offer to trade ...", "rejected trade
  offer", completed trades — that we inspected on 2026-09-29 (about 56k
  rows over 45 games, about 1000 "traded" rows). We saw no board layout or
  per-player hand state in those tables; whether full move/state logs
  exist elsewhere in the release is **unverified**.
- With only 60 human games, Dobre & Lascarides extracted per-action-type
  preferences to bias planning rather than cloning a policy: "we had only 60
  games rather than millions", and the preferences "provide useful
  information in significantly improving the planning agent"; a
  conditioned-type-preference POMCP agent won 34.75% against 3 Stac agents
  (25% is chance; the unconditioned variant got 25.63%, not significantly
  different at p<0.01) — [POMCP with Human Preferences in Settlers of
  Catan, AIIDE
  2018](https://cdn.aaai.org/ojs/13014/13014-52-16531-1-2-20201228.pdf) —
  takeaway: tiny human corpora help as priors. Their 60-game corpus was
  "collected by an omniscient server" (cited there as Afantenos et al.
  2012); it is not stated to be the same set as STAC's 45 annotated games,
  and we did not verify that it is public.
- The same work is the subject of a thesis: M. S. Dobre, *Low-resource
  learning in complex games*, University of Edinburgh, 2019 — [ERA
  record](https://era.ed.ac.uk/handle/1842/35534) — takeaway: a longer
  treatment of the same low-data approach (abstract only checked).
- Human data would not fix the current bottleneck: cloning `HeuristicAgent`
  on 3000 games reached only 50-55% masked accuracy on the spatial blocks
  with a ~8 point train/val gap, which log 004 attributes to task
  difficulty rather than data scarcity, so the lever there is
  [008](008-board-aware-encoder.md) — no external source; observed in catan
  log
  [004](https://github.com/guidodinello/catan/blob/main/docs/experiments/004-bc-warm-start.md).
- The human games planned in catan experiment 007 are evaluation data, not
  training data: 24 games with a single human, designed to compare bots
  against chance ([018](018-human-baseline-sanity-check.md)) — no external
  source; observed in catan log 007.

## Availability and licensing

| Source | Usable for training? | Notes |
|---|---|---|
| colonist.io | No, without written permission | Terms above; the extension export does not change this. |
| STAC | Yes, non-commercial, share-alike | 45 games; event text and chat; move/state completeness unverified. |
| Dobre & Lascarides 60-game corpus | Unknown | Public release not verified. |
| Real-life tournaments | Nothing found | Negative search result: searches on 2026-09-29 found no public move-level records of tournament games (catan.com [championships](https://www.catan.com/catan-fans/championships) lists events, not game records). Not evidence that none exist. |

## How to test

- **Step 0 (free):** check exactly what the STAC release contains
  (situated tables, JSON, game graphs) and whether trade offers, their
  responses and the resources held at the time can be reconstructed.
- **Metric:** if step 0 passes, an opponent-acceptance model (offer,
  context) -> accepted, scored by held-out log-loss against a base-rate
  predictor; downstream, win rate in a trade-aware benchmark against
  trading opponents (#28) with and without the model, at equal budget
  (see [020](020-modular-trade-agent.md) stage iii).
- **Gate:** held-out log-loss beats the base rate, and the downstream win
  rate has a non-overlapping Wilson interval
  ([005](005-eval-statistics.md)).
- **Cost estimate:** low for step 0; moderate for the acceptance model. Not
  worth scheduling before trading opponents exist.

## Result

Not yet attempted.

**Motivated by:** catan issue
[#28](https://github.com/guidodinello/catan/issues/28) (an agent that
learns to trade) and the bottleneck analysis in catan log
[004](https://github.com/guidodinello/catan/blob/main/docs/experiments/004-bc-warm-start.md).

## Related notes

- [002 — Behavior-cloning warm start before PPO](002-bc-warm-start.md)
- [008 — Board-aware encoder / spatial inductive bias](008-board-aware-encoder.md)
- [012 — Enable the reserved trade heads](012-trade-heads.md)
- [018 — Human baseline as a sanity check for learned / hand-written agents](018-human-baseline-sanity-check.md)
- [020 — Modular agent: separate trade module over a strategy policy](020-modular-trade-agent.md)
