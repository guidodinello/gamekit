# 011 — KL guard vs the previous snapshot

**Status:** running
**Last touched:** 2026-10-05

## Hypothesis

An automated guard comparing the current policy against a recent snapshot
(by KL divergence, or by the entropy/clip-fraction signals that
accompanied a real collapse) could catch a self-play collapse in progress
and stop or roll back training before the run's time budget is wasted —
but the guard has to watch for the collapse that actually happened, not
the one PPO's own tuning guidance warns about.

## Why we believe it

- PPO's own adaptive-KL-penalty variant reduces the penalty coefficient
  when divergence between old and new policy is *below* a target and
  increases it when *above* — a mechanism built around watching KL move
  in either direction — [Schulman et al. 2017, *Proximal Policy
  Optimization Algorithms*](https://arxiv.org/abs/1707.06347) — takeaway:
  treating KL divergence as a signal worth actively watching (not just
  logging) is an established PPO pattern, even though Schulman et al.
  found the *clipped* objective outperforms the KL-penalty variant as the
  optimization method itself.
- truco-py's collapse is the concrete motivating case, and its actual
  signature is the **opposite** of a KL spike: `approx_kl` went to `0.0`
  and `clip_fraction` to `0` as the policy froze into a degenerate
  loop (`ep_len_mean` 2 → 165, `ep_rew_mean` 0.4 → -7.28,
  `entropy_loss` -0.14 → -1.18e-05) — no external source; observed in
  truco-py `logs/train_selfplay_gpu.log` and `docs/session-2026-06-06.md`.
  A guard that only fires on a KL *increase* would have missed this
  collapse entirely — it needs to watch for the policy going flat
  (entropy/KL/clip-fraction all collapsing toward zero together), not only
  for divergence blowing up.

## Refinement (2026-09-27)

- **What we'd actually build is a policy-freeze detector, not a KL
  guard.** SB3's `approx_kl` measures how far the old policy moved to
  the new one *within one update*, i.e. step size. It is not a KL
  against an earlier snapshot. The collapse we've seen shows up as the
  step size and entropy going flat together, so the first thing to build
  reads metrics SB3 already logs. A true snapshot KL (masked KL between
  the live policy and a checkpoint N chunks back, on a fixed probe set
  of observations) is a later phase, only needed if phase 1 misses a
  collapse. The filename keeps "KL guard" for link stability.
- **Calibrated against the real collapse.** truco-py commits its June
  run's SB3 console output (`logs/train_selfplay_gpu.log`): 143 PPO
  updates of 8,192 steps each, from 5.71M to 6.89M total steps. Healthy
  updates 0–60 sit at `abs(entropy_loss)` 0.14–0.195 and `clip_fraction`
  0.008–0.020. Candidate rules, replayed offline (no external source;
  computed from that log):

  | Rule | Fires at update | Total steps |
  |---|---|---|
  | `abs(entropy_loss)` **and** `clip_fraction` both < 50% of their trailing-20-update median, 3 consecutive updates | 74 | 6,324,224 |
  | same, < 25% of trailing median | 126 | 6,758,400 |
  | absolute near-zero (`abs(entropy_loss)` < 1e-3, `clip_fraction` < 1e-3, `approx_kl` < 1e-4) | 124 | 6,742,016 |
  | (manual kill, per truco-py log 005) | — | ~6.89M |

  The 50% trailing-median rule fires about **420k steps (~52 updates)
  before** "everything is zero" and doesn't fire on the healthy
  prefix. The 25% variant fires late for an instructive reason: a
  trailing median follows a gradual decline downward, so a tight ratio
  never trips until the very end.
- **Why trailing, not a frozen early baseline:** catan's healthy v1 run
  lowered `entropy_loss` from −0.287 to −0.168 over the whole run
  (catan log 002), a 41% drop. A baseline frozen at the start with a
  50% threshold would be uncomfortably close to firing on normal
  policy sharpening. A trailing window allows slow decline and catches
  fast decline, and fast decline is what the collapse looks like.
- **Why this complements `RegressionGuard` instead of duplicating it:**
  catan's `RegressionGuard` looks at the win rate every 250k steps and
  needs `patience=2` consecutive qualifying evals, so it can't stop
  sooner than ~500k steps after a real drop. It also sees noisy n=200
  evals (see [009](009-longer-runs-and-resume.md)'s refinement). The
  detector looks at every 8,192-step update and needs 3 of them.
- **Dropped from scope: "a KL anchor to damp catan log 004's
  oscillation."** That band is flat within noise (χ²=11.7, df=11,
  p=0.39; see 009), so there's no oscillation to damp. A KL *penalty*
  toward the BC clone is also a regularizer, not a guard. If it's worth
  pursuing, it gets its own note with its own motivation.

## How to test

**Build (split along gamekit's framework boundary):**

- **gamekit:** a framework-free `PolicyFreezeDetector` (e.g. in
  `gamekit.rl`) with `observe(entropy, clip_fraction, approx_kl) -> bool`.
  Its parameters are `window=20`, `ratio=0.5`, `patience=3`, plus a
  warm-up so it can't fire before the window fills. Unit-test it against
  the 143-row trajectory above, stored as a small CSV fixture parsed
  from truco-py's log: it must fire at update 74 and must not fire on
  updates 0–60. Add synthetic negative controls: slow linear decay,
  noise around a constant, and a single-update dip.
- **Consumers (catan first, then truco-py):** an SB3 callback that
  feeds the detector once per update. It reads `train/entropy_loss`,
  `train/clip_fraction` and `train/approx_kl` from
  `self.logger.name_to_value` in `_on_rollout_end`. In SB3's
  `OnPolicyAlgorithm.learn` loop, the previous update's `train/*`
  values are recorded but not yet dumped at that point. Confirm that
  ordering with a test against the pinned sb3 2.9 before relying on it.
  On a fire, the callback returns `False` from the next `_on_step`, and
  `rl/train.py` treats it like a `RegressionGuard` stop
  (`stopped_early=True`, best checkpoint kept).

- **Metric:** detection lead time (steps between the detector firing and
  the absolute-near-zero point) and the false-fire count on healthy runs.
- **Gate:**
  1. *Offline, positive:* fires at least 300k steps before the absolute
     near-zero point on truco-py's June trajectory. The calibration
     above gives ~420k.
  2. *Offline, negative:* zero fires when catan logs 002–004's
     tfevents are replayed through it (`rl_runs/tb/`, gitignored, so
     this depends on them still being on disk), and on truco-py's
     healthy prefix.
  3. *Online:* on a collapse triggered on purpose, the detector stops
     the run within 3 updates of the threshold being crossed and before
     `RegressionGuard` would have. Either
     [010](010-entropy-schedule.md)'s 010a clean-pool control, if it
     collapses, or a deliberately contaminated run (an *unscoped*
     `OpponentPool` over a directory holding truco-py's collapsed
     `truco_selfplay_final.zip`, the mechanism
     [013](013-selfplay-pool-contamination.md) validated) serves as the
     positive case.
- **Cost estimate:** gates 1 and 2 need no training: about half a day
  for the gamekit detector, fixture and tests, plus the catan callback.
  Gate 3 costs one ~1M-step truco-py run, which can be the same run as
  010a's control.

## Result

Not yet implemented in either consumer repo; the detector's offline
calibration above is a design input, not a result. This note is a response to
[005](005-eval-statistics.md)'s finding that truco-py's only stopping rule
was a manual, coarse "3 consecutive checkpoints < 80%" check.

**Linked from** truco-py logs
[004](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/004-april-selfplay-collapse.md)
(verdict: *inconclusive for that specific run* — the KL/clip_fraction
trace is inferred by analogy to the June run, not independently measured
from the April tfevents) and
[005](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/005-june-threshold-mix-collapse.md)
(verdict: *supports the note, with a caveat the note itself needs to
carry* — the confirmed measurement is `approx_kl → 0.0` and
`clip_fraction → 0`, the opposite of a spike; any guard built from this
note has to watch for the collapse-to-zero signature, not only a rise, as
already reflected in this note's Hypothesis and How-to-test sections
above). **Motivated by** catan log
[003](https://github.com/guidodinello/catan/blob/main/docs/experiments/003-selfplay-v2-baseline-mix-0.5.md),
which names this note as a higher-effort control deliberately deferred
from that PR's scope, without implementing or testing it.

**Motivated by** catan log
[004](https://github.com/guidodinello/catan/blob/main/docs/experiments/004-bc-warm-start.md)
as well: its follow-up section names "a KL guard against the clone's own
initial policy, which might damp the 10.5-20% oscillation seen here" as an
untried direction, without implementing or testing it — a narrower proposal
than this note's "guard vs the previous snapshot," anchoring the guard to
the BC clone's starting policy specifically rather than a rolling recent
checkpoint.

**Result from truco-py log 009 (2026-10-05), an online negative control:**
2026-10-05: status moved from `planned` to `running` (interim evidence only).
The paragraph at the top of this Result, "Not yet implemented in either
consumer repo", is true of gamekit and is kept as history; it is no longer true
of truco-py. truco-py log [009](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/009-retrain-mixed-pool.md) ([PR #33](https://github.com/guidodinello/truco-py/pull/33)) shipped
`PolicyFreezeDetector` in its own `training/collapse.py` and fed it once per PPO
update in both arms (`training/run.py` builds it with the defaults: window 20,
ratio 0.5, patience 3, plus the absolute near-zero fallback). **It never fired
in either arm over 20M steps each, and neither arm collapsed** (M and C both
ended at 91-92% vs Threshold). No external source; observed in the log.

What this is and is not: two healthy runs with zero false fires, an online
negative control. It is not gate 2 (offline replay on catan logs 002-004), and
not gate 3 (a collapse triggered on purpose), so it says nothing about
detection lead time. The gamekit-side `PolicyFreezeDetector` is still not built.

## Related notes

- [001 — Self-play opponent mix vs a fixed baseline](001-self-play-opponent-mix.md)
- [002 — Behavior-cloning warm start before PPO](002-bc-warm-start.md)
- [005 — Eval statistics: Wilson intervals and eval-in-loop](005-eval-statistics.md)
- [009 — Longer runs / resume when the curve has not bent](009-longer-runs-and-resume.md)
- [010 — Entropy schedule instead of a fixed coefficient](010-entropy-schedule.md)
- [013 — Self-play pool contamination across runs](013-selfplay-pool-contamination.md)
