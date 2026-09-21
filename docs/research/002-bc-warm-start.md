# 002 — Behavior-cloning warm start before PPO

**Status:** planned
**Last touched:** 2026-09-20

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

## Why this is `planned`, not `validated`

Three things keep this from being a validated result yet:

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

Not yet run for any `gamekit` consumer game. truco-py's BC run exists but
does not answer the question above (see "Why this is `planned`").

## Related notes

- [003 — Discount horizon vs episode length](003-discount-horizon.md)
