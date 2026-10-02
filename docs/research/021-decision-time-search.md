# 021 — Decision-time search for Catan: ISMCTS with the trained policy/value network as priors

**Status:** tested (search helps with a known opponent model, an upper bound; self-model confirmation pending)
**Last touched:** 2026-10-02

## Hypothesis

A short, determinized Information Set MCTS (ISMCTS) at decision time —
using the trained catan policy as the move prior and its critic as the
leaf evaluator, AlphaZero-style and with no retraining — raises the win
rate against 3 `HeuristicAgent`s above the current best of 20.72% and may
close the 25% gate that pure model-free training has not cleared.

## Why we believe it

### Context correction: who searches how

The "strongest chess engine" and "MCTS with a network" are two different
lines of work, and this note borrows from the second one.

- Stockfish is an alpha-beta engine, not an MCTS one. Its own release post
  says the evaluation is "used in alpha-beta (PVS) search to find the best
  move" — [Introducing NNUE
  Evaluation](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)
  — takeaway: the network replaces the *evaluator*, the search stays
  alpha-beta. The Chess Programming Wiki lists Principal Variation Search
  among its search features and notes that "Stockfish 16, released June 30,
  2023, removes the classical evaluation from the engine and focuses on
  NNUE neural networks" — [Stockfish - Chess Programming
  Wiki](https://www.chessprogramming.org/Stockfish).
- MCTS with network priors is the AlphaZero line: "Instead of an alpha-beta
  search with domain-specific enhancements, AlphaZero uses a general-purpose
  Monte-Carlo tree search (MCTS) algorithm", guided by learned "move
  probabilities and value estimates" — Silver et al. 2017, [Mastering Chess
  and Shogi by Self-Play with a General Reinforcement Learning
  Algorithm](https://arxiv.org/abs/1712.01815) (quotes from the paper body,
  not the abstract) — takeaway: policy = prior over moves, value = leaf
  evaluation. That is the recipe this note proposes to reuse.
- Leela Chess Zero follows it: "the PUCT used in AGZ and Lc0 replaces
  rollouts (sampling playouts to a terminal game state) with a neural
  network that estimates what a rollout would do" (AGZ = AlphaGo Zero) —
  [Technical Explanation of Leela Chess
  Zero](https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/)
  — takeaway: no rollouts to the end of the game; the critic stands in.

### Current state: neither game searches

- catan's `RLAgent.choose_action` calls `model.predict(obs,
  action_masks=...)` once per atom and composes the atoms into an action; there
  is no lookahead — no external source; observed in
  [`catan/agents/rl_agent.py`](https://github.com/guidodinello/catan/blob/main/agents/rl_agent.py).
- truco-py's `RLAgent.choose_action` likewise calls `self._model.predict` on
  a MaskablePPO checkpoint — no external source; observed in
  [`truco-py/agents/rl_agent.py`](https://github.com/guidodinello/truco-py/blob/main/agents/rl_agent.py).
- Search appears only in passing: catan's
  [`engineering-review.md:141`](https://github.com/guidodinello/catan/blob/main/docs/research/engineering-review.md)
  quotes catanatron's advice about cheap state copies for MCTS rollouts,
  and truco-py's
  [`pro-agent-roadmap.md`](https://github.com/guidodinello/truco-py/blob/main/docs/pro-agent-roadmap.md)
  (Option 3) covers CFR/DeepCFR — no external source; observed in those
  files.

### Why plain MCTS does not transfer

Catan has dice, hidden hands and dev cards, and four players; textbook MCTS
assumes none of these.

- **Dice (chance).** "Unlike classic games such as Chess and Go, stochastic
  game trees include chance nodes in addition to decision nodes. How MCTS
  should account for this added uncertainty remains unclear", and "another
  way is to simply sample a single outcome when encountering a chance node.
  This is common practice in MCTS when applied to stochastic games" —
  Lanctot et al. 2013, [Monte Carlo
  \*-Minimax Search](https://www.ijcai.org/Proceedings/13/Papers/093.pdf)
  (IJCAI) — takeaway: sampling dice outcomes is the cheap default, at the
  cost of variance.
- **Hidden information.** ISMCTS searches trees of information sets
  instead of trees of game states — Cowling, Powley & Whitehouse 2012,
  [Information Set Monte Carlo Tree
  Search](https://eprints.whiterose.ac.uk/id/eprint/75048/) (IEEE TCIAIG) —
  takeaway: the abstract confirms the information-set approach and that
  the paper studies MCTS for hidden information; I could not fetch the
  full text, so nothing here rests on its strategy-fusion discussion.
- **Determinization's known flaws, and why it often works anyway.** Perfect
  Information Monte Carlo (PIMC) — sample a hidden state, search it as if
  it were public — "has been criticized in the past for its theoretical
  deficiencies" (strategy fusion and non-locality, identified by Frank &
  Basin 1998) "but in practice it has often produced strong results in a
  variety of domains" — Long et al. 2010, [Understanding the Success of
  Perfect Information Monte Carlo Sampling in Game Tree
  Search](https://ojs.aaai.org/index.php/AAAI/article/view/7562) (AAAI) —
  takeaway: whether determinization is adequate is game-dependent and
  measurable (leaf correlation, bias, disambiguation factor); Catan has
  not been measured on those. Frank & Basin's own paper is cited only
  through Long et al.; I could not fetch it.
- **Four players.** Multi-player UCT is "nearly identical to regular UCT.
  ... The only difference ... is that in line 5 the average score for
  player p is used instead of a single average payoff for the state" —
  Sturtevant 2008, [An Analysis of UCT in Multi-Player
  Games](https://webdocs.cs.ualberta.ca/~nathanst/papers/mpuct_icga.pdf)
  (ICGA Journal) — takeaway: each node keeps a per-seat value, and a critic
  used as leaf evaluator must supply one value per seat.

### Catan MCTS track record

- Szita, Chaslot & Spronck 2010, [Monte-Carlo tree search in Settlers of
  Catan](https://research.tilburguniversity.edu/en/publications/monte-carlo-tree-search-in-settlers-of-catan)
  (ACG 2009, LNCS 6048): per the abstract (seen only in a web-search
  snippet — Springer requires a login), they "apply MCTS to the
  multi-player, non-deterministic board game Settlers of Catan" and report
  "considerable playing strength when compared to game implementation with
  existing heuristics" — takeaway: plain MCTS works in Catan against
  heuristic bots, with no numbers I can cite.
- Dobre & Lascarides 2018, [POMCP with Human Preferences in Settlers of
  Catan](https://cdn.aaai.org/ojs/13014/13014-52-16531-1-2-20201228.pdf)
  (AIIDE): POMCP with a factored belief that tracks each player's
  resources and dev cards independently. Against 3 Stac agents ("a
  hand-coded decision tree" that "was the state of the art player" — not
  JSettlers), Table 3 gives plain POMCP 33.45% at 10k iterations and
  47.94% at 40k (25% is equal strength); their best variant reaches 40.70%
  and 53.65%. Their tournament table shows the neural-net-seeded variant
  (POMCP-NN) at 34.89% against Stac versus 33.45% for the uniform-prior
  baseline — takeaway: search beats a heuristic bot in Catan by a wide
  margin, but their iteration counts and per-decision times (seconds) are
  far above what the benchmark here can afford, their opponent is not our
  `HeuristicAgent`, and a learned prior helped little in their setup.

## Proposed design (sketch, not a spec)

- Search runs over composed actions; the prior for an action is the product
  of atom probabilities under `ActionComposer`'s prefix mechanism (see
  [004](004-factored-action-head.md)).
- Each iteration determinizes the opponents' hands and dev cards
  consistently with public information (Dobre-style independent tracking is
  the simplest belief model); dice outcomes are sampled.
- Leaves are scored by the critic, one value per seat, backed up per seat
  (multi-player UCT); no rollouts to game end. **Score leaves with
  V+ = V + own public VP / 10, not raw V** (catan log
  [009](https://github.com/guidodinello/catan/blob/main/docs/experiments/009-critic-calibration.md): raw V carries a negative own-VP term and is negatively
  rank-correlated with own VP; see Result).
- Trades stay masked, as today ([012](012-trade-heads.md),
  [020](020-modular-trade-agent.md)).

## How to test

- **Metric:** win rate vs 3 `HeuristicAgent`s, n=4000, seat-rotated, with
  Wilson intervals and a two-proportion test
  (`gamekit.mc.testing.two_proportion_test`) against the no-search arm, per
  [005](005-eval-statistics.md). Both arms use the same checkpoint
  (`catan_bc_ft_long_10031616`, the +8M point of catan log
  [006](https://github.com/guidodinello/catan/blob/main/docs/experiments/006-longer-run.md):
  829/4000 = 20.72% [19.50%, 22.01%]) and a fixed per-move budget
  (simulations; also report ms). Re-run the no-search baseline in the same
  session rather than reusing 006's number: 20.72% is the best of five
  fixed points (the +10M gate point was 19.90%), so it is slightly
  winner's-curse biased.
- **Also report:** mean and p95 decision latency, and how win rate scales
  across two or three budgets (a curve, not one point).
- **Gate:** significantly above the no-search baseline; to close the
  Phase 5 gate, the CI must lie entirely above 25%.
- **Cost estimate:** measured engine throughput is 4922 env-steps/s
  single-process, at 1584 engine steps and 450 learner steps per 4-player
  episode (catan
  [PR #17](https://github.com/guidodinello/catan/pull/17)). With the policy
  in the loop, catan log
  [005](https://github.com/guidodinello/catan/blob/main/docs/experiments/005-gpu-inference.md)
  measured roughly 570-770 fps end to end, so the network forward, not the
  engine, is likely the search bottleneck. Illustratively, 4000 games x
  ~450 decisions is ~1.8M search decisions, so every simulation per move
  adds ~1.8M engine expansions plus network evaluations to one benchmark.
  This is a back-of-envelope figure; re-measure the cost of a state copy
  and of a batched critic call before choosing a budget.

## Risks

- **Throughput.** As above; the per-move budget may end up too small to
  change decisions at all.
- **State copies.** Catan's state is not a flat array; copying is the known
  cost in catanatron's experience (engineering-review.md:141).
- **Dice as chance nodes** add variance; sampled outcomes need many
  simulations to average out.
- **The critic** was trained on on-policy states under a shaped reward
  (potential-based VP, gamma 0.999), so it is neither calibrated as a win
  probability nor reliable off-distribution, and search deliberately visits
  off-policy states. Measured once, on decision states, in catan log
  [009](https://github.com/guidodinello/catan/blob/main/docs/experiments/009-critic-calibration.md): V+ ranks well (AUC about 0.83) but weakly early in
  the game (about 0.69), it is not a probability, and sibling differences
  are small (see Result). Still check it on *searched* states, which log
  009 did not cover.
- **Four players, non-zero-sum.** Modeling opponents with our own policy
  may be a poor fit for `HeuristicAgent`.
- **The prior may already be near-greedy.** Search adds little if the
  policy's argmax rarely changes; log how often search overrides it.

## Truco: CFR, not MCTS

Truco hides cards and rewards bluffing, so a determinized search that
treats the sampled deal as public will not bluff or reason about what its
own bets reveal. That is reasoning, not a sourced result: I found no
source I could fetch that states it directly. What the sources do support
is that CFR is the standard tool for this class: "Counterfactual Regret
Minimization (CFR) is the leading framework for solving large
imperfect-information games", and Deep CFR "use[s] deep neural networks to
approximate the behavior of CFR in the full game" — Brown et al. 2019,
[Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164)
(ICML); CFR itself is from Zinkevich et al. 2007, [Regret Minimization in
Games with Incomplete
Information](https://papers.nips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)
(NIPS). Recommendation: for truco, follow truco-py's
[`pro-agent-roadmap.md`](https://github.com/guidodinello/truco-py/blob/main/docs/pro-agent-roadmap.md)
Option 3 (CFR/DeepCFR) rather than this note's ISMCTS; that option is not
duplicated here.

## Result

**Search measured:** catan log
[011](https://github.com/guidodinello/catan/blob/main/docs/experiments/011-decision-time-search.md)
(catan [PR #45](https://github.com/guidodinello/catan/pull/45), merged as
`72cb62e`; issue [#41](https://github.com/guidodinello/catan/issues/41)).
Same checkpoint `catan_bc_ft_long_10031616`, vs 3 `HeuristicAgent`,
seat-rotated, on 006's boards (seeds 1..n), no retraining. No external
source; observed in the log.

- **A1, search with a known opponent model (`heur`, S=128 simulations per
  searched decision): 34.62% [33.17, 36.11] vs A0 (no search) 20.72%
  [19.50, 22.01], n=4000 each, +13.9 pts** (z=13.89; exact McNemar: 770
  boards won only with search, 214 only without). A0 reproduced 006's
  829/4000 exactly. **This is an upper bound**: the opponents inside the
  search are the same deterministic `HeuristicAgent` that sits at the
  table, so search predicts their moves exactly and is uncertain only
  about hidden cards and dice. It does not transfer to other opponents.
- **A2, the transferable number (`self`, opponents inside the search =
  the agent's own greedy policy, S=64): 27.2% [24.53, 30.04], n=1000,
  +5.6 pts** over 21.6% without search on the same boards (p=0.0036).
  Exploratory (about 5 pts MDE at n=1000, half the A1 budget), and the
  interval's lower bound is below 25%.
- **Budget curve** (`heur`, same 1000 boards, S=0/32/64/128): 21.6% ->
  26.3% -> 29.8% -> 35.3%, monotone with no sign of saturation. Post-hoc,
  `heur` S=64 vs `self` S=64 is -2.6 pts for not knowing the opponents
  (p=0.20), which n=1000 cannot resolve.
- **Latency** (per searched decision, 16 concurrent workers on a loaded
  laptop): S=128 `heur` mean 845 ms, p95 1.5 s; S=64 `self` mean 1.8 s,
  p95 3.8 s, max 45 s. 53% of top-level decisions have one legal action
  and are not searched. Fine for an offline benchmark, slow for
  interactive play, so it is a benchmark role only.
- **The prior is not near-greedy.** Search overrides the greedy action in
  9.3% of searched decisions (MAIN 13.1%, first settlement 12.4%); about
  one searched decision in eleven carries the +13.9 pts. Also, an
  `EndTurn` leaf's per-sample noise is about 3.5x the spread between
  sibling moves, so a larger budget mostly buys noise reduction.
- **Design departures from the sketch above** (log 011): (1) the backup is
  single-seat with a fixed opponent model, not per-seat multi-player UCT,
  because opponents are simulated rather than searched, so per-seat leaf
  values were never needed; (2) the tree is open-loop over the rl seat's
  own top-level decisions (opponent moves are simulated, not tree nodes);
  (3) Q is min-max normalized over the tree (MuZero's `MinMaxStats`) with
  V+ leaves and +-1 terminals, and an unvisited child takes the parent's
  mean normalized Q. Trades stay masked, as sketched.
- **Phase 5 gate:** A1's Wilson lower bound (33.17%) is above 25%, so the
  pre-registered rule is met **only with the known-opponent-model
  qualifier**. Whether that counts toward closing the gate is Guido's
  call and is still undecided. The model-free 20.72% is unchanged.
- **Next:** catan issue
  [#46](https://github.com/guidodinello/catan/issues/46) is the planned
  confirmation: `self` opponent model, S=128, n=4000 on fresh boards, to be
  pre-registered in a new log before any measured run (none exists yet). Its
  gate is the self arm's Wilson lower bound above 25% **and** a significant
  win over no search, which would be the model-free route to the Phase 5
  gate that A1 cannot settle.
- Caveats: one checkpoint; the determinizer is weaker than card counting
  (it ignores what a steal revealed).
- If a second game wants it, the torch-free search core
  (`agents/ismcts.py` in catan) could move into gamekit; truco does not,
  since this note recommends CFR there.

**Prerequisite measured (the critic as a leaf evaluator):** catan log
[009](https://github.com/guidodinello/catan/blob/main/docs/experiments/009-critic-calibration.md) (catan [PR #43](https://github.com/guidodinello/catan/pull/43); n=4000 seat-rotated games per arm,
checkpoint `catan_bc_ft_long_10031616`). It tests the evaluator, not
search. No external source; observed in the log.

- **Evaluate leaves on V+ = V + own public VP / 10, not raw V.** The
  shaping telescopes, so V is roughly `E[gamma^N * (+-1)] - phi(s)` with
  `phi` = public VP / 10: raw V carries a negative own-VP term (rank
  correlation with own VP -0.15 to -0.18). AUC vs win, one random decision
  per game: V+ 0.832 [0.816, 0.847] (vs 3 `Heuristic`) and 0.830 [0.814,
  0.845] (vs 3 `TradingHeuristic`); raw V 0.768 and 0.760; public VP lead
  alone 0.777 and 0.774.
- **V+ is not a probability.** It is monotone in the win rate (about 1% to
  63% across deciles), and (V+ + 1) / 2 tracks that rate within about
  0.11, but is systematically off (inside the Wilson interval in 7 of 10
  deciles vs `Heuristic`, 4 of 10 vs `TradingHeuristic`). Use it to
  rank, and do not back up a "win probability" from it.
- **Weak early, good late.** AUC of V+ by the seat's own turn: about
  0.69-0.70 in the first 12 turns, 0.82-0.83 for turns 12-23, 0.86 from
  turn 24. Within a fixed public state (own VP x lead) it is only about
  0.67; it adds about +0.04 AUC over a logistic model of lead, own VP and
  turn. So a shallow search from early states is leaning on the weakest
  part of the critic.
- **Sibling differences are small:** dV between sibling states (accept vs
  offer withdrawn, 56,408 offers) has 5th-95th percentile about -0.009 to
  +0.013, against V+ decile means that span about -1 to +0.2. Exploration constants
  and any "prefer the higher child" margins must be set on that scale.
- **Evidence on unseen states is partial.** The probe states (opponent's
  turn, after a simulated trade) were never learner decision points, yet V
  behaved sensibly there (receiving cards raised V in 92% of offers,
  paying without receiving in 8%). V read at the raw pending-offer state,
  with its never-seen offer features, shifted about 3x more than dV. V was
  read only from the `rl` seat's own view: the per-seat leaf values
  multi-player UCT needs (other seats' views) were not measured.
- Caveat: one checkpoint, decision states only, none of it on states
  produced by a search.

**Motivated by:** Guido's question of why neither game searches at decision
time, and the correction that Stockfish is alpha-beta + NNUE rather than
MCTS.

## Related notes

- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
- [008 — Board-aware encoder / spatial inductive bias](008-board-aware-encoder.md)
- [009 — Longer runs / resume when the curve has not bent](009-longer-runs-and-resume.md)
- [012 — Enable the reserved trade heads](012-trade-heads.md)
- [019 — Human Catan game data as a training source](019-human-catan-game-data.md)
- [020 — Modular agent: separate trade module over a strategy policy](020-modular-trade-agent.md)
