# 022 — League ratings: anchored Bradley-Terry/Elo over a round-robin, with the win matrix alongside

**Status:** running
**Last touched:** 2026-10-10

## Hypothesis

Ranking agents by one anchored Bradley-Terry/Elo scale fitted to a seat-rotated
round-robin, reported **next to** the raw pairwise win matrix and a cycle check,
ranks checkpoints and rule bots more faithfully than the one-opponent win rates we
use today, and exposes the rock-paper-scissors structure a single rating hides.
Tracked by [gamekit#38](https://github.com/guidodinello/gamekit/issues/38); the
module is `gamekit.league`.

## Why we believe it

### Why a single opponent is not enough (our own evidence)

- truco-py log 008 (n=4000 matches per pairing, seat-rotated, Wilson CIs) measured
  the same checkpoints against two opponents and got opposite orderings. The
  threshold-trained `thr5M` beats Threshold 80.1% [78.8%, 81.3%] but Random only
  53.7% [52.2%, 55.2%]; the random-trained `random_final` is the reverse, 51.0%
  [49.5%, 52.6%] vs Threshold and 66.0% [64.5%, 67.5%] vs Random. The Threshold
  agent itself beats Random only 52.2% [50.7%, 53.7%], so "vs Random" is a weak
  yardstick, and the threshold-trained agents "are exploiting Threshold far more than
  they beat Random" — no external source; observed in truco-py
  [log 008](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/008-seat-rotated-rebenchmarks.md).
  A rating from either column alone ranks `thr5M` and `random_final` in opposite order.
- Caveat, from the same log: those numbers come from the **pre-fix engine** (rules
  audit findings A-01..A-11, voided hands up to 2.1 per 100 matches) and "are not
  rules-correct skill estimates". They motivate the design; they are not the first
  league result. The first truco league is meant to run after the engine fixes
  (truco-py PR #14, issues #7-#13) — no external source; observed in log 008.

### Pairwise ratings: Bradley-Terry, Elo, TrueSkill

- Bradley-Terry models the probability that `i` beats `j` from per-agent strengths,
  and MM (minorization-maximization) iterations fit it by maximum likelihood.
  Hunter's paper presents MM algorithms for generalized Bradley-Terry models and
  states conditions under which each converges to the unique maximum likelihood
  estimator — [Hunter 2004, *MM algorithms for generalized Bradley-Terry models*,
  Annals of Statistics
  32(1)](https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-1/MM-algorithms-for-generalized-Bradley-Terry-models/10.1214/aos/1079120141.full)
  — takeaway: the fit is an iteration with a convergence guarantee, which is why
  `gamekit.league.ratings` can do it in a few lines of stdlib Python with no numpy.
  **Unverified verbatim:** the title, author and year are confirmed (Crossref), but
  Project Euclid blocked a direct fetch and Crossref carries no abstract, so the
  convergence statement above is a paraphrase of a fetch summary, not a quote, and I
  did not read the existence condition. The code enforces the usual one (the win
  graph must be strongly connected) with a `ValueError` at `prior_draws=0`; that is
  my own reading of the model. The original [Bradley & Terry
  1952](https://www.jstor.org/stable/2334029) is on JSTOR; the fetch returned only a
  site error page, so it is **not** relied on here.
- Elo is the same logistic model fitted online: "Choosing learning rate η = 16 or 32
  recovers the updates introduced by Arpad Elo", and Elo ratings are at a stationary
  point under batch updates "iff the matrices of empirical probabilities and
  predicted probabilities have the same row-sums" (Proposition 1) — [Balduzzi,
  Tuyls, Perolat & Graepel 2018, *Re-evaluating
  Evaluation*](https://arxiv.org/abs/1806.02643) (NeurIPS; quotes from the PDF body,
  section 2.1, not the abstract) — takeaway: that is the Bradley-Terry score equation
  (each agent's observed total score equals its predicted total), so a batch BT-MLE
  fit and "Elo" are the same predictor. We fit the batch MLE (order-independent,
  with CIs) and report it on the Elo scale (`400 * log10(strength ratio)`).
- TrueSkill adds an uncertainty per player and handles teams and draws; it keeps a
  mean and an uncertainty per player, treats a team's skill as the sum of its
  players' skills, and counts close performances as draws — [TrueSkill ranking
  system](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
  (Microsoft Research) — takeaway: it is the right tool for a ladder with few games
  per player and many players. **Unverified verbatim:** this is a paraphrase of a
  fetch summary; the page was unreachable by a direct request, so no quote is made.
  A closed round-robin of tens of agents with thousands of games per pairing has no
  such sparsity, so we take the batch BT fit with bootstrap CIs and leave TrueSkill out.

### Why anchor

- Strengths are identified only up to a common scale, so a rating means nothing
  until it is pinned. OpenAI Five does it with a fixed reference pool: it evaluates
  "by comparing them to a pool of fixed reference agents with known skill using the
  TrueSkill rating system. In our TrueSkill environment, a rating of 0 corresponds
  to a random agent" — [OpenAI et al. 2019, *Dota 2 with Large Scale Deep
  Reinforcement Learning*](https://arxiv.org/abs/1912.06680) (quote from the PDF,
  section on evaluation) — takeaway: this is the design in issue #38 (Random = 0).
  `gamekit.league` pins a named reference agent to `anchor_rating` (default 0) and
  re-anchors after every bootstrap refit, so the anchor's own interval is exactly
  `[0, 0]` and every other interval is uncertainty **relative to the anchor**.
- An Elo rating "can be inflated by instantiating many copies of an agent it beats"
  (Balduzzi et al. 2018, same PDF) — takeaway: an open roster drifts. Cross-run
  comparability needs a fixed anchor and a roster that is stated with the result;
  `gamekit.league` derives each pairing's seeds from `(seed, a, b)` only, so adding
  an agent later never changes an existing pairing.
- Small samples and perfect scores need a prior. The BayesElo documentation says it
  "uses a prior distribution over ratings, that increases the likelihood that the
  ratings of players are close to each other", and a player with a perfect 10-0
  is rated 169 Elo by BayesElo against 300 by Elostat, in a two-player table where the
  loser gets the mirror image (a 338-point versus a 600-point gap) — [Coulom, *Bayesian
  Elo*](https://www.remi-coulom.fr/Bayesian-Elo/) — takeaway: without a prior, a
  10-0 pairing has no finite rating. The page does **not** describe how its prior
  is built, so ours is not a reimplementation: `gamekit.league` adds
  `prior_draws=1` virtual game (half a win each way) per played pairing. That only
  keeps the fit finite; it is much weaker than BayesElo's, and a 10-0 result still
  gives a 529-point gap (BayesElo's is 338) with a bootstrap interval of [213, 529] (observed by running the
  module). Treat few-game pairings with the Wilson interval in the matrix, not the
  rating.

### Non-transitivity: why the win matrix must accompany the ratings

- "Elo bakes-in the assumption that relative skill is transitive; but Elo is
  meaningless – it has no predictive power – in cyclic games", and "rock, paper and
  scissors will all receive the same Elo ratings. Elo's predictions are p̂ij = 1/2
  for all i, j" — Balduzzi et al. 2018 (PDF body) — takeaway: this is exactly our
  rock-paper-scissors unit test, where the three ratings come out equal while the
  matrix has 90% edges. It is also why: the 0.9/0.1 matrix has the same row sums as
  the all-0.5 prediction, so equal ratings are an exact fixed point of the fit, and
  only the per-pairing residual in the matrix shows the misfit.
- Real games are mostly transitive with a cyclic tail: "their geometrical structure
  resemble a spinning top, with … cycles that exist at a particular transitive
  strength", measured over nine two-player zero-sum games including Go and StarCraft
  II — [Czarnecki et al. 2020, *Real World Games Look Like Spinning
  Tops*](https://arxiv.org/abs/2004.09468) — takeaway: expect a single rating to
  rank most of a league correctly and fail among agents of similar strength, which is
  where our checkpoints and rule bots are. Quotes checked against the abstract page;
  I did not read the paper body, so none of its quantitative claims are used.
- `gamekit.league` therefore always returns the raw matrix with Wilson intervals
  (`gamekit.mc.wilson_interval`), the BT-predicted rate and the residual per pairing,
  and every 3-cycle `A > B > C > A` among **significant** edges only: an edge needs
  a Wilson interval entirely above 50%, otherwise noise alone manufactures cycles.

### Leagues in prior work (pointer to #39 only)

- AlphaStar's league uses the pairwise win probabilities as its sampling input. Self-play
  "may chase cycles (for example, where A defeats B, and B defeats C, but A loses to
  C) indefinitely without making progress"; main agents use "a prioritized fictitious
  self-play (PFSP) mechanism that adapts the mixture probabilities proportionally to
  the win rate of each opponent against the agent", sampling opponent `B` with
  probability `f(P[A beats B]) / sum_C f(P[A beats C])` — [Vinyals et al. 2019,
  *Grandmaster level in StarCraft II using multi-agent reinforcement
  learning*](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
  (Nature; DeepMind's unformatted PDF; the Nature page itself redirected to a cookie
  check and was not read) — takeaway: the matrix this module produces is the input
  PFSP needs. The DeepMind [blog
  post](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
  says only that the league extends "fictitious self-play to a group of agents"; it
  has no sampling details.
- OpenAI Five "play[s] the latest policy against itself for 80% of games, and play[s]
  against older policies for 20% of games" (same OpenAI paper, rollout section) —
  takeaway: the simpler, unprioritized form of the same idea.
- **Not implemented here.** Using the league to choose training opponents is
  [gamekit#39](https://github.com/guidodinello/gamekit/issues/39); it depends on this
  note's module and is out of scope for this one.

### Multi-player games (catan, 4 seats)

- **The reduction.** A pairing `(a, b)` in an `N`-seat game fields the alternating
  lineup `(a, b, a, b, …)`, so each agent holds `N/2` seats; the pairwise outcome is
  "which agent's seat won", and a seat may be a team (truco-py's 2-team game uses
  `num_seats=2`). Under a Luce first-choice model, where the winner among the seated
  agents is chosen with probability proportional to strength, `P(a wins) =
  (N/2)·π_a / ((N/2)·π_a + (N/2)·π_b) = π_a / (π_a + π_b)`. This is exactly
  Bradley-Terry, and two identical agents sit at 50%. This is my own derivation under
  that model assumption, not a sourced result; the check is the test
  `identical_agents_in_a_four_seat_game_rate_equal`.
- **Why not 1-vs-field.** `a` against `N-1` copies of `b` puts parity at `1/N` (25% in
  catan). A pairwise fit fed "`a` won 25%" would rate identical agents unequally. Rank-based
  Plackett-Luce over finishing order is the principled generalization and is future
  work.
- **Odd `N > 2`** cannot be split evenly: `gamekit.league` raises
  `NotImplementedError` rather than guessing.
- **Limit (reasoning, not a sourced claim).** Catan copies of an agent do not
  cooperate, so a 2v2 rating gap says how the agents rank against each other in a
  mixed table, not what `a` wins against three `b`s. Catan's 1v3 gate benchmarks
  (`HeuristicAgent`, `TradingHeuristic`) stay as they are; the league adds the common
  scale. If 2v2 and 1v3 orderings disagree, that is a finding about the game, not a bug.
- **Ties.** A game with no winning seat is dropped from the fit and the Wilson
  denominators and reported as `ties` per pairing.

## How to test

- **Metric:** anchored Elo with a 95% bootstrap CI per agent, the Wilson-CI win
  matrix, and the cycle report, over a seat-rotated round-robin at n per pairing
  (a multiple of the seat count, as `gamekit.benchmark.run_arm` requires).
- **Gate for the module (done in `tests/test_league_*.py`):** BT recovers known
  strengths; the anchor is pinned and shifting it only translates; bootstrap CIs cover
  the truth in at least 85% of 60 synthetic replications; a rock-paper-scissors
  matrix gives equal ratings and one detected cycle; a resumed run equals an
  uninterrupted one; identical agents in a 4-seat game rate equal.
- **Gate for the first uses (done; see Result):** truco — a league over the 008 checkpoints plus
  Random and Threshold anchors, **after** the engine fixes, run overnight on the
  homelab HP; catan — rank `catan_bc_ft_long`, `rl_search` (catan log 011),
  `Heuristic` and `TradingHeuristic`. State the roster and the anchor with every result.
- **Cost estimate:** `k` agents need `k(k-1)/2` pairings. The issue's truco roster is
  log 008's 12 checkpoints plus Random and Threshold, 14 agents and 91 pairings; at
  4000 matches each and the ~24 matches/s a laptop smoke test measured vs Threshold
  (log 008), that is about 4.2 hours single-process. RL-vs-RL pairings may be
  slower than that figure, so budget overnight on the HP. Results are
  one JSON per pairing via `gamekit.results`, so an interrupted run resumes by pairing.

## Result

Not yet run. The module and its tests are in [gamekit#38](https://github.com/guidodinello/gamekit/issues/38);
the first league results will be linked from here once the consumer repos have
logs.

**Added 2026-10-05, truco-py log 009** ([log](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/009-retrain-mixed-pool.md), [PR #33](https://github.com/guidodinello/truco-py/pull/33)): no
league has been run, so this note's status is unchanged. But the
motivation above cites Threshold beating Random only 52.2%; 009's Phase 0
re-measured it on the rebuilt engine at **87.5% [86.5, 88.5]** (n=4000,
seat-rotated), so that figure is void and "vs Random" is saturated, not weak.
009 therefore treats vs-Random as descriptive and uses a held-out VonNeumann as
its second yardstick, which is the single-opponent problem this note addresses.
No external source; observed in the log.

**First use, truco-py log 010 (2026-10-07):** status moved from `planned` to
`running` (interim evidence only: the truco first use is done, the catan one is
not). The "Not yet run" paragraph above is kept as history; it is no longer true.
truco-py log
[010](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/010-league-009-checkpoints.md)
([PR #35](https://github.com/guidodinello/truco-py/pull/35), merged as `1c52ab0`)
ran the first league on `gamekit.league` (gamekit 0.3.0 at `76c364b`): 9 agents
(the seven log 009 checkpoints M5/M10/M20/C5/C10/C20 and bc_init, plus Threshold
and Random), 36 pairings, n=10000 each, `num_seats=2` team slots, `prior_draws=1`,
1000 bootstrap resamples, anchor **Threshold = 0**. Two departures from this note's
plan: the roster is the 009 checkpoints, not the 008 ones (#26 invalidated those),
and the anchor is Threshold, not Random = 0 as issue #38 proposed. Descriptive: log
010 pre-registered that no verdict is scored for this note, so none is scored here.
No external source; observed in the log, and every number below was checked
against its `league.json`, except where a bullet names another source.

- **Ratings (Elo, 95% bootstrap CI):** M20 353.2 [348.9, 357.2], M10 333.4
  [329.6, 337.2], C20 308.3 [304.4, 312.2], C10 305.4 [301.3, 309.3], M5 298.9
  [294.9, 302.9], C5 285.4 [281.5, 288.9], Threshold 0, bc_init -13.3 [-17.1, -9.7],
  Random -171.7 [-175.9, -167.2]. No ties in any pairing, no significant 3-cycle.
  Every Wilson half-width is under 1 point.
- **M beats C head to head at every matched step:** M5 vs C5 54.8%, M10 vs C10
  57.0%, M20 vs C20 58.9%. VonNeumann is not in this roster, so this does not
  contradict log 009's vs-VonNeumann ordering.
- **What the matrix shows that the rating hides.** The four largest
  Bradley-Terry residuals are Random cells (Random vs Threshold 12.7% against a
  predicted 27.1%; vs bc_init 17.0% vs 28.7%; vs M5 15.6% vs 6.2%; M20 vs Random
  86.7% against a predicted 95.4%); the fifth is bc_init vs Threshold. The two lines treat Random differently: the M
  line beats it less often than it beats Threshold (M20 86.7% vs 92.5%), the C line
  more often (C5 94.1% vs 88.3%). And M5 beats C10 (52.1%) and C20 (51.4%) head to
  head although it rates below both.
- **Seat rotation and reproducibility.** Seat split 49.9 / 50.1 in C20 vs M20 (from
  that pairing file). Per the log, a C20-vs-M20 cross-check at n=200 was identical
  (wins, ties, config hash, winners sha256) on the laptop (torch cu128) and the HP
  (torch cpu).
- **Cost, per the log:** 6 h 47 m on the HP with 3 workers; the 21 RL-vs-RL
  pairings dominate, and the HP measured about 7x slower per core than the laptop.
- **What the API did well:** pairing seeds and the config hash do not depend on the
  roster, so the parallel workaround below writes the same files a full run would;
  resume refuses a pairing with a different hash and the stamps could be re-derived
  from the files; `summarize_league` is pure over the files, so re-summarizing needs
  no replay; pairing files are written to a temp name and renamed.
- **Where it was awkward.** `run_league` is serial, so 3 workers took one 2-agent
  `run_league(agents=[a, b], anchor=a, n_bootstrap=10)` call per pairing in a process
  pool; `anchor` and the small `n_bootstrap` only serve the summary each call
  produces as a side effect. That side effect is a race: each call writes
  `league.json` with a plain write, last writer wins, and the content is a 2-agent
  summary. The in-run copy was discarded and `league.json` regenerated from the 36
  pairing files. Pairing files were not affected.
- **Not exercised yet:** the N>2 seat reduction (catan), ties, and the prior at small n. (Superseded for the seat reduction by catan log 013 below; ties and the small-n prior were still not exercised there.)
- **Next:** the catan first use. A run-pairing API plus an atomic `league.json`
  write would remove the workaround:
  [gamekit#47](https://github.com/guidodinello/gamekit/issues/47).

**Catan first use, catan log 013 (2026-10-10):** the N>2 seat reduction, the other
first use. catan log
[013](https://github.com/guidodinello/catan/blob/main/docs/experiments/013-first-league.md)
([results PR #65](https://github.com/guidodinello/catan/pull/65), pre-registration
[#52](https://github.com/guidodinello/catan/pull/52); same driver as truco log
[010](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/010-league-009-checkpoints.md))
ran `gamekit.league` at gamekit `05f271e` with **11 agents** (8 RL checkpoints: bc_clone,
bc_ft_2m, long_4m / long_8m / long_10m, and the three log 010 runs ent_ctrl / ent_ramp /
ent_decay; plus heuristic, trading_heuristic, random), 55 pairings, **n=4000 each (220,000
games)**, `num_seats=4`, lineup `(a, b, a, b)`, `prior_draws=1`, 1000 bootstrap
resamples, anchor **heuristic = 0**. Departures from the plan above: no `rl_search` agent
(too slow for the HP), and the roster is wider than the four agents the issue named.
Descriptive: log 013 pre-registered that no verdict is scored for this note, so none is
scored here. No external source; observed in the log, and the numbers below are from its
`league.json` and pairing files.

- **The 4-seat reduction ran end to end.** Every one of the 55 pairing files has 4000 games
  and a distinct config hash, each agent occupied each seat 2000 times (only `abab` / `baba`
  occur, so `aabb` seatings are not tested), and re-running `summarize` on the pairing files
  reproduced `league.json` exactly. The reduction itself is still the Luce-model derivation
  above, not a sourced result.
- **Ratings (Elo, 95% bootstrap CI):** trading_heuristic +16.2 [+10.6, +21.2], heuristic 0
  (anchor), long_10m -62.9 [-67.6, -57.8], long_8m -69.2 [-74.4, -64.2], ent_decay -89.9
  [-95.0, -84.6], ent_ctrl -116.1 [-121.0, -111.2], long_4m -123.4 [-128.9, -118.4], ent_ramp
  -132.3 [-137.6, -127.2], bc_clone -161.1 [-166.4, -156.1], bc_ft_2m -178.8 [-183.8,
  -173.6], random -687.3 [-696.2, -676.6]. No RL checkpoint rates above the heuristic.
  `heuristic` beats `long_10m` 59.8% in 2v2, which agrees in sign with the 1v3 benchmark
  (long_10m 20.72% against 25% parity); the two numbers are on different scales, so only the
  sign is compared.
- **Ties: 0 of 220,000 games, so the tie path was not exercised in a run** (only in the CI test
  of the all-ties case, where the pairing file is written and the fit then refuses a league
  with no decisive game).
- **The small-n prior was not exercised.** n=4000 per pairing; the closest cell is random vs
  trading_heuristic (2 wins in 4000), and `prior_draws=1` is negligible there.
- **Bradley-Terry fits the RL block poorly.** gamekit reports **5 significant 3-cycles, all
  through long_10m** (for example ent_ctrl > long_10m > long_8m > ent_ctrl). The largest
  residual is long_10m vs long_8m: **66.7% observed against 50.9% fitted** (+0.158); next,
  long_10m vs ent_ramp (48.4% against 59.9%) and vs ent_ctrl (47.3% against 57.6%). long_10m
  beats the log 006 line (long_4m, long_8m) and the bc_* checkpoints by more than one scalar
  per agent predicts, and does worse against the log 010 runs. As in truco 010, the
  win matrix shows what the rating hides; the single rating is an adequate summary against
  the baselines and a poor one inside the RL block.
- **Seat rule caveat.** RL seats always reject trade offers, so trading_heuristic's +16 is
  not trade skill against a trading opponent, and two trading_heuristic copies can trade
  with each other, which breaks the no-cooperation premise above in its cells. Per log 013
  this is a bias of the roster, not of the module.
- **Reproducibility, partial.** The cross-machine check (3 pairings, n=200, laptop torch
  cu128 vs HP CPU torch) matched on the per-game winner hashes in all three; the laptop's
  record-digest hashes for two pairings and its `config_hash` files were lost in a reboot
  (outputs were in `/tmp`), so the pre-registered criterion was only partly checked. The
  league's own stamps (config hash, wins + ties = n, seat occupancy) were re-derived from
  the files.
- **Cost, per the log:** 46.5 h on the HP with 3 workers (1.29x the 35.9 h smoke estimate,
  which was based on n=8 per pairing): about 64 min per RL-vs-RL pairing (28 of them), 40
  min per RL-vs-baseline (24), 14 min per baseline-vs-baseline (3).
- **What the API did well / where it was awkward.** Resumability by pairing and
  roster-independent seeds and hashes worked as in truco 010. Instead of truco's
  one-`run_league`-per-pairing pool (whose `league.json` writes race), catan ran one serial
  `run_league` whose `play` callback fans each pairing's games out to a persistent
  3-worker pool: no race, a correct `league.json` from the run itself, and a barrier at the
  end of each pairing that was negligible at n=4000. That workaround is still a workaround
  for [gamekit#47](https://github.com/guidodinello/gamekit/issues/47) (a public `run_pairing`
  and an atomic `league.json` write), which stays open.
- **Next:** a trading RL agent (catan #28) before trade skill can sit on this scale; a
  rank-based Plackett-Luce reduction for 4 seats, given how the single scale fits the RL
  block, remains the future work named above. Both first uses are now recorded.

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
- [013 — Self-play pool contamination across runs](013-selfplay-pool-contamination.md)
- [016 — Positional (mano) advantage must be rotated out of a benchmark arm](016-positional-advantage-rotation.md)
- [021 — Decision-time search for Catan](021-decision-time-search.md)
