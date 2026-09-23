# 002 — Behavior-cloning warm start before PPO

**Status:** running
**Last touched:** 2026-09-23

## Hypothesis

Cloning a scripted agent's action distribution with supervised learning
before starting PPO gets the policy to a reasonable starting point faster
than a random-weight cold start.

## Why we believe it

- DAgger and the broader imitation-learning line of work formalize
  training a policy to match an expert's actions as a starting point (or
  full substitute) for RL — [Ross, Gordon & Bagnell 2011, *A Reduction of
  Imitation Learning and Structured Prediction to No-Regret Online
  Learning*](https://arxiv.org/abs/1011.0686) — takeaway: imitation
  learning is a well-studied way to bootstrap a policy, with known
  failure modes (distribution shift) worth being aware of when the clone
  target is a scripted agent rather than a human expert.
- AlphaGo's original pipeline trained a supervised policy network from
  human expert games before any reinforcement learning — [Silver et al.
  2016, *Mastering the game of Go with deep neural networks and tree
  search*](https://www.nature.com/articles/nature16961) — takeaway:
  SL-then-RL is a proven pattern, not specific to board games with human
  data — AlphaStar's league likewise started "only with agents trained by
  supervised learning" before self-play took over.
- truco-py actually ran this: cloning `ThresholdAgent` (its scripted,
  MC-threshold-derived agent) with cross-entropy over 650,312 `(obs,
  action)` pairs, 10 epochs, reached **92.0% training-set action
  accuracy** in **1050.1s** (17.5 minutes) — no external source; observed
  in truco-py `logs/bc_pretrain.log` and `README.md:117`.

## Why truco-py's run didn't validate this

Three things kept truco-py's own BC run from settling this hypothesis:

- **92.0% is training-set accuracy, not held-out.** truco-py's own plan
  set the bar as "BC accuracy ≥ 70% on held-out `ThresholdAgent` actions",
  but `scripts/pretrain_bc.py` never implements a train/val split — the
  accuracy is computed over the same data the model was fit on. The
  held-out number this hypothesis actually needs has never been measured.
- **"In minutes" doesn't match the run.** The plan estimated ~5 minutes;
  the actual run took 17.5 minutes (`Done in 1050.1s`).
- **The "replaces a 5M-step PPO warm-up" claim is untested, and the plan
  contradicts itself on the number it's replacing** — one line says BC
  "achieves in minutes what PPO warm-start takes ~50M steps to
  accomplish" (`docs/plan_rl_agent.md:121`), another says it "replaces the
  need for a 5M-step PPO warm-start" (`:276`). No side-by-side ablation
  (cold-start PPO vs BC-init PPO, same step budget) exists anywhere in the
  repo.

## How to test

- **Metric:** held-out action-prediction accuracy against the scripted
  agent (a genuine split, unlike truco-py's run); then win rate of
  BC-initialized PPO vs cold-start PPO at matched step budgets.
- **Gate:** BC-init PPO should reach a given win-rate threshold in
  meaningfully fewer steps than cold-start PPO, with overlapping-CI
  checkpoints ruling out a lucky seed.
- **Cost estimate:** one BC pre-training run (cheap, ~20 min based on
  truco-py's measurement) plus two PPO runs at the same step budget
  (cold-start baseline + BC-initialized) — the marginal cost is the
  duplicate PPO run for the baseline.

## Result

catan is the first `gamekit` consumer to actually run this on a genuine
held-out split. catan `docs/experiments/004-bc-warm-start.md` (PR #21) cloned
`HeuristicAgent` by cross-entropy, splitting 3,000 generated games by game id
(not by sample) into 456,707 train / 51,893 validation examples, and measured
**87.4% val masked top-1 accuracy** — the held-out number this note's "Why
truco-py's run didn't validate this" section says was never measured
anywhere before. Val masked accuracy varied sharply by atom block: the
spatial-placement blocks (`vertex_settlement` 51.6%, `edge` 55.4%, `hex`
50.5%) were far weaker than the rest (`resource` 94.8%, `simple` 94.4%,
`vertex_city` 76.8%, `steal` 76.2%, `opener` 78.8%).

The clone alone (no RL) scored **10.0% [8.76%, 11.39%]** at n=2000 vs 3
`HeuristicAgent`s — well below the 25.0% [24.34%, 25.68%] a *perfect* clone
would reach by symmetry, despite the strong overall held-out accuracy.
Fine-tuning the clone with the project's self-play PPO recipe
(`lr=1e-4`, `ent_coef=0.01`, `baseline_mix=0.5`, 3,000,000 steps) reached
**16.2% [15.09%, 17.37%]** vs 3 `HeuristicAgent`s and **93.5% [92.69%,
94.22%]** vs 3 `RandomAgent`s at n=4000 — non-overlapping with catan log
003's cold-start ceiling of 12.075% [11.10%, 13.12%] / 90.025% [89.06%,
90.92%] on both fronts, using fewer total steps (3M fine-tune vs 003's ~7.5M
self-play total). The 25% gate (CI must exclude 25% from below) is still not
met: 17.37% < 25%.

**Why this stays `running`, not `validated`:** log 004's own verdict is
"validated (technique-level), gate not met" for its narrower claim, but the
matched ablation this note's "How to test" section actually asks for — a
budget-matched cold-start-vs-BC-init comparison, same step count, same
hyperparameters — still does not exist. Log 004 is explicit that its own
comparison isn't that ablation: 003 received far more total compute (~7.5M
steps) than this run's 3M-step fine-tune, and used different `lr`/`ent_coef`
(3e-4/0.02 vs 1e-4/0.01) besides, so "a higher number here is not 'PPO learns
faster from a clone,' only 'this config, from this start, cleared this
control.'"

**Open problem this run surfaces:** BC fidelity on the spatial placement
decisions (vertex/edge/hex, 50-55% masked accuracy) is the bottleneck the
fine-tune inherited, not something the fine-tune introduced — see
[008](008-board-aware-encoder.md).

**Tested by:** catan log
[004](https://github.com/guidodinello/catan/blob/main/docs/experiments/004-bc-warm-start.md),
the first held-out-split BC result in the project — technique validated at
the level of "beat the cold-start control," gate not met.

truco-py log
[002](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/002-bc-warm-start.md)
remains as before: verdict *inconclusive*, confirms at the file/line level
that its 92.0% figure is training-set accuracy: `scripts/pretrain_bc.py:132-135`
builds a single `TensorDataset`/`DataLoader` over the whole 650,312-pair
collection, with no train/val split anywhere in the file, and the same
batches the optimizer just updated on are what `accuracy` is scored against
inside the same epoch loop.

## Related notes

- [003 — Discount horizon vs episode length](003-discount-horizon.md)
- [008 — Board-aware encoder / spatial inductive bias](008-board-aware-encoder.md)
