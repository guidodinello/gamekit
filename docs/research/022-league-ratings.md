# 022 — League ratings: anchored Bradley-Terry/Elo over a round-robin, with the win matrix alongside

**Status:** planned
**Last touched:** 2026-10-01

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
- **Gate for the first uses (pending):** truco — a league over the 008 checkpoints plus
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

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
- [013 — Self-play pool contamination across runs](013-selfplay-pool-contamination.md)
- [016 — Positional (mano) advantage must be rotated out of a benchmark arm](016-positional-advantage-rotation.md)
- [021 — Decision-time search for Catan](021-decision-time-search.md)
