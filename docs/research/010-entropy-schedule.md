# 010 — Entropy schedule instead of a fixed coefficient

**Status:** idea
**Last touched:** 2026-09-23

## Hypothesis

Scheduling the entropy coefficient (e.g. higher early to keep exploring,
decayed later to let the policy commit) should behave better than a single
fixed coefficient for the whole run, particularly for recovering from or
avoiding an entropy collapse.

## Why we believe it

- PPO's objective includes an entropy bonus specifically to encourage
  exploration by penalizing low entropy — [Schulman et al. 2017,
  *Proximal Policy Optimization
  Algorithms*](https://arxiv.org/abs/1707.06347) — takeaway: the entropy
  coefficient is a first-class knob in the objective, not an
  implementation detail, so it's a reasonable place to look when training
  degenerates toward a deterministic policy too early.
- A large-scale empirical study of on-policy RL design choices treats
  entropy-related settings as one of the dimensions worth tuning
  systematically rather than leaving at a library default — [Andrychowicz
  et al. 2020, *What Matters In On-Policy Reinforcement
  Learning?*](https://arxiv.org/abs/2006.05990) — takeaway: there's
  empirical precedent for entropy-coefficient choices mattering enough to
  be worth a dedicated sweep, which a schedule is one way to explore.
- truco-py proposed exactly this in response to its collapse (entropy
  going to near-zero within ~900k steps) — raising `ent_coef` from 0.01 to
  0.05–0.1 for the self-play phase — but never implemented it: the `docs`
  note itself says the change "Requires adding a CLI flag (`--ent-coef`)
  to `train.py`", and that flag does not exist in the repo — no external
  source; observed in truco-py `docs/session-2026-06-06.md` (Option C).

## How to test

- **Metric:** `entropy_loss` over training (does it stay above some floor
  instead of collapsing to ~0), and win rate vs the fixed-coefficient
  baseline.
- **Gate:** a scheduled/higher coefficient should avoid the entropy
  collapse this note is a response to, without a corresponding drop in win
  rate.
- **Cost estimate:** low to implement (a schedule or a higher constant),
  moderate to test (needs at least one full run through the point where
  the fixed-coefficient baseline previously collapsed).

## Result

Not yet attempted — the CLI flag this would require was never built in
either consumer repo.

**Linked from** truco-py log
[005](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/005-june-threshold-mix-collapse.md)
(verdict: *untested* — the session's proposed `--ent-coef` flag was never
built and no entropy-schedule run exists in that repo). **Motivated by**
catan log
[003](https://github.com/guidodinello/catan/blob/main/docs/experiments/003-selfplay-v2-baseline-mix-0.5.md),
which names this as an untried direction its own win-rate ceiling
motivates, without testing it.

**Motivated by** catan log
[004](https://github.com/guidodinello/catan/blob/main/docs/experiments/004-bc-warm-start.md)
too: its BC-warm-start fine-tune used a fixed `ent_coef=0.01` from step 0
(deliberately gentler than 003's 0.02, chosen to protect the clone from
early entropy-driven collapse) and still oscillated 10.5-20.0% for the
entire run — the log's own follow-up section names "an entropy schedule
instead of the fixed 0.01 used here" as an untried direction, without
testing it.

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [002 — Behavior-cloning warm start before PPO](002-bc-warm-start.md)
- [011 — KL guard vs the previous snapshot](011-kl-guard.md)
