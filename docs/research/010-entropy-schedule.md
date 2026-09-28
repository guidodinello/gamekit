# 010 — Entropy schedule instead of a fixed coefficient

**Status:** planned
**Last touched:** 2026-09-27

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

## Refinement (2026-09-27)

This note had one hypothesis that turns out to cover two separate claims
with different evidence. Split them:

- **010a — collapse prevention.** A higher or floored coefficient keeps
  entropy from going to ~0. The motivating case is truco-py's June run.
  But truco-py log 005 and [013](013-selfplay-pool-contamination.md)
  already put that collapse down to **stale-checkpoint pool
  contamination**, which run-scoped pools fix (gamekit 0.3.0; truco-py
  had not adopted it as of log 005). So 010a is only testable once the
  pool is clean. If a clean-pool run with `ent_coef=0.01` doesn't
  collapse, there's nothing for a schedule to prevent. That wouldn't
  reject the note. It would make it moot for truco-py.
- **010b — plateau escape.** A schedule gets a higher final win rate
  than a constant, on a run that is *not* collapsing. That's catan's
  situation: catan log 002's TensorBoard diagnostic found no collapse
  (`entropy_loss` −0.287 → −0.168, `approx_kl` 0.002–0.004), and the
  plateaus in logs 003/004 are flat within noise (see
  [009](009-longer-runs-and-resume.md)'s refinement). The prior for
  010b is weaker than for 010a. Nothing we've observed points at
  entropy as catan's bottleneck; catan log 004 points at BC fidelity
  on spatial decisions instead.
- **The direction of a schedule depends on the starting point.** The
  usual "high early, decay later" suits a cold start. catan log 004
  chose a *low* early `ent_coef` (0.01) specifically so it wouldn't wash
  out a near-deterministic BC clone. For a warm start, the matching
  schedule is closer to "low early, then raise to a floor." The arms
  below keep the two apart.
- **Stale claim corrected.** The CLI flag the Result section describes
  as missing now exists in catan (`rl/train.py --ent-coef`, a fixed
  constant). truco-py still hardcodes `ent_coef=0.01`
  (`training/train.py`) and has no flag.

### Implementation constraints (found in catan `rl/train.py`)

- **Key the schedule on absolute `model.num_timesteps`, not SB3's
  `progress_remaining`.** catan calls `model.learn()` once per
  250k-step eval chunk, so anything driven by `progress_remaining`
  restarts every chunk. `rl/train.py` already guards against the same
  sawtooth for `lr_schedule` on the `--bc-init` path. Keying on
  cumulative steps also makes a schedule survive `--resume` unchanged.
- **Apply it with a callback, not the constructor.** SB3's `PPO` reads
  the float `self.ent_coef` inside each `train()` call. A
  `BaseCallback._on_rollout_start` that sets
  `self.model.ent_coef = schedule(self.model.num_timesteps)` updates it
  once per rollout (8,192 steps at catan's `n_steps=512 × 16 envs`),
  with no chunk-boundary artifacts. Record the value it set (e.g.
  `train/ent_coef_effective`) so the effective schedule shows up in
  TensorBoard, not just the CLI.
- **The callback also avoids the resume trap in
  [009](009-longer-runs-and-resume.md).** `--resume` silently keeps the
  pickled `ent_coef`, but a callback overwrites it before the first
  update.
- **Where the code lives:** the schedule itself is a pure
  `step -> float` function (piecewise-linear over
  `(step, value)` breakpoints). It could go in `gamekit.rl` so catan and
  truco-py share one tested implementation. The SB3 callback stays in
  each consumer, following gamekit's no-framework-imports split. This
  is optional: a local function in catan is enough to run 010b.

## How to test

### 010a — truco-py, collapse prevention (run first: sharper gate)

**Prerequisites (truco-py):** adopt gamekit ≥0.3.0 with
`OpponentPool(run_id=...)`; add `--ent-coef` and a `--ent-schedule`
flag wired to the callback above; export the per-update SB3 metrics
(`approx_kl`, `clip_fraction`, `entropy_loss`) to a CSV or tfevents in
the run directory, as [011](011-kl-guard.md) also needs.

**Protocol:** replicate truco-py log 005's self-play config
(`--opponents selfplay --shaped-reward --threshold-mix 0.5 --n-envs 16`,
`--load checkpoints/truco_threshold_5000000.zip`), 1.5M self-play steps
per arm. log 005's collapse started ~550k steps into self-play and was
complete by ~1.1M. Same seed across arms.

1. **Clean-pool control, `ent_coef=0.01`.** If this doesn't collapse
   within 1.5M steps, stop: 010a is moot for truco-py. Record that as
   further evidence for 013, and move to 010b.
2. **Only if the control collapses:** the arms are constant 0.05 and
   `0.05 → 0.02` linear over the first 1M steps.

- **Metric:** `abs(entropy_loss)` relative to its median over the first
  100k self-play steps, plus the n=1000 win rate vs `ThresholdAgent` at
  the 1.5M checkpoint.
- **Gate:** an arm passes if `abs(entropy_loss)` never falls below 25% of
  its early median for 3 consecutive updates (the [011](011-kl-guard.md)
  detector) **and** its n=1000 win rate is not significantly below the
  control's best pre-collapse checkpoint (`two_proportion_test`,
  p ≥ 0.05). At p≈0.85 and n=1000, differences under ~4.5 points can't
  be resolved, so "no drop" means "no drop this benchmark can see."
- **Cost estimate:** one 1.5M-step run (the control) if the answer is
  "moot", three if not. The truco-py flag and callback are about half a
  day.

### 010b — catan, plateau escape (lower prior)

**Prerequisites (catan):** a `--ent-schedule` flag parsed into
breakpoints and applied by the callback above; `--ent-coef` stays as
the constant shorthand.

**Protocol:** catan log 004's fine-tune config, from the same BC clone,
3M steps, same seed, one run per arm:

- *control:* constant 0.01 (log 004 itself; re-benchmark its 3M
  checkpoint at n=4000 if [009](009-longer-runs-and-resume.md) hasn't
  already)
- *warm-start ramp:* 0.005 → 0.02 over the first 1M steps, then constant
- *cold-start decay, for reference:* 0.02 → 0.005 over 3M

- **Metric:** n=4000 win rate vs 3 `HeuristicAgent` at the fixed 3M
  checkpoint (not the in-loop best; see 009's winner's-curse point),
  plus the `entropy_loss` trace.
- **Gate:** an arm is *validated* if it beats the control at 3M with
  `two_proportion_test` p < 0.05 after a Bonferroni correction for two
  arms (p < 0.025). The n=4000 minimum detectable difference is about
  2.3 points. If neither arm clears it, 010b is *rejected* for catan's
  BC fine-tune at this budget.
- **Cost estimate:** two 3M-step runs (~90 min each at `--envs 8`, per
  log 004) plus three n=4000 benchmarks. About half a day of catan
  code.

## Result

Not yet attempted. catan has since gained a constant `--ent-coef`
flag, but neither consumer repo has a schedule (see Refinement).

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
- [009 — Longer runs / resume when the curve has not bent](009-longer-runs-and-resume.md)
- [011 — KL guard vs the previous snapshot](011-kl-guard.md)
- [013 — Self-play pool contamination across runs](013-selfplay-pool-contamination.md)
