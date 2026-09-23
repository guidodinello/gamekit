# 017 — Measure the Amdahl ceiling and the real batch size before building an inference server

**Status:** validated
**Last touched:** 2026-09-23

## Hypothesis

A centralized batched inference server only pays off when (a) the component it
speeds up has a wall-time share large enough to give a worthwhile Amdahl
ceiling, and (b) requests to it actually coalesce into batches once it's
running end-to-end. A micro-benchmark of batched forward-pass latency in
isolation is evidence of neither — it says nothing about how much of the real
pipeline's time that component occupies, and nothing about whether concurrent
callers will actually arrive close enough together to be served in a batch.

## Why we believe it

- Amdahl's law formalizes the first half directly: if a component makes up a
  fraction `s` of total (serial) execution time, speeding up *only* that
  component — even infinitely — bounds the whole program's possible speedup
  to `1/(1-s)` — [Amdahl, G. M. 1967, *Validity of the single processor
  approach to achieving large scale computing
  capabilities*](https://doi.org/10.1145/1465482.1465560), AFIPS '67
  (Spring), p. 483 — takeaway: before building a faster version of a
  component, the component's measured share of wall time already caps the
  payoff, independent of how good the faster version turns out to be.
- `multiprocessing.connection.wait(object_list, timeout=None)` "Wait[s] till
  an object in object_list is ready. Returns the list of those objects in
  object_list which are ready. If timeout is a float then the call blocks for
  at most that many seconds" — [Python docs,
  `multiprocessing.connection.wait`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.connection.wait)
  — takeaway: `timeout` is a maximum wait, not a minimum collection window —
  the call returns as soon as *anything* is ready, so a server that calls
  `wait()` once and processes whatever comes back will batch only requests
  that happened to already be ready at that instant, not requests that arrive
  during the timeout period.
- catan measured both halves directly and found neither one held for its
  opponent-inference server. Profiling the CPU pipeline (`rl/profile.py`)
  found opponent-checkpoint inference at 25.2-25.8% of worker wall time at
  the default `baseline_mix=0.5` across pools of 1, 4 and 12 checkpoints (and
  39.1% at the `baseline_mix=0.0` worst case) — capping the best possible
  end-to-end speedup at `1/(1-0.258) ≈ 1.35x` (`1.64x` worst case), nowhere
  near truco-py's reported 8.8x. A CPU/CUDA latency micro-benchmark of the
  same forward pass (`agents.rl_agent.lean_predict_batch`) looked
  promising in isolation — 1.4 µs/sample on GPU at batch size 64 vs. 17.8
  µs/sample on CPU, ~13x — but instrumenting the running server's actual
  batch sizes (`InferenceServer.stats()`) found a mean group size of
  1.04-1.28 regardless of pool size, because `_collect_batch`'s single call
  to `wait(conns, timeout)` returns on the first ready connection rather than
  collecting a window of arrivals. End-to-end FPS reflected this: pool1 (1
  checkpoint, the best case for batching) went from 637 to 701 fps (+10%,
  mean group size 1.28); pool14 (12 checkpoints) stayed flat, 602-626 fps
  either way (mean group size 1.04) — no external source; observed in catan
  `docs/experiments/005-gpu-inference.md`.
- A fix was attempted (draining newly-ready connections for the rest of the
  wait window instead of returning on the first signal) and reverted: it
  turned the timeout into a real collection window, but every sparse request
  (the common case, given the measured group sizes above) now paid close to
  the full window as pure added latency, collapsing throughput to roughly
  `1/_BATCH_WAIT_S` requests/second server-wide and never completing a
  training chunk that previously finished in ~30s — no external source;
  observed in catan `docs/experiments/005-gpu-inference.md`.

## How to test

- **Metric:** the component's measured share `s` of total wall time from a
  profiler pass (before writing any server code), and — once a server
  exists — its actual mean batch size, instrumented as a count rather than
  inferred from throughput (a count is contention-independent; a wall-clock
  FPS number is not).
- **Gate:** `1/(1-s)` has to be worth the build in the first place, and the
  server's measured mean batch size has to be clearly greater than 1
  end-to-end, not just in an isolated forward-pass benchmark — both
  conditions have to hold, since either one alone is not sufficient (catan's
  case had `s` low enough to bound the win at 1.35x *and* a batching bug that
  meant even that ceiling was never approached).
- **Cost estimate:** low — a profiler pass (catan's `rl/profile.py` is
  reusable as a pattern: worker-side split, main-process split, a latency
  micro-bench) plus one counter added to the server's request-collection
  loop.

## Result

catan `docs/experiments/005-gpu-inference.md` (issue #20, PR #22): the
opponent-checkpoint inference server was **rejected** as a training-throughput
win. Amdahl capped it at ~1.35x regardless of implementation; the
implementation itself never approached even that ceiling, batching at a mean
group size of 1.04-1.28 because of the `wait()` semantics above. The two
components this note's reasoning doesn't touch — the PPO update step and BC
pre-training — were adopted as opt-in CUDA options instead, because neither
depends on batching requests across processes: the update step measured 2.4x
on the update itself (~12% end-to-end, matching Amdahl at its ~21% wall-time
share) and BC pre-training measured ~23x on the compute step, both verified
bit-identical to CPU output on real data.

**Tested by:** catan log
[005](https://github.com/guidodinello/catan/blob/main/docs/experiments/005-gpu-inference.md)
is the negative result this note generalizes from.

**Linked from:** truco-py log
[005](https://github.com/guidodinello/truco-py/blob/main/docs/experiments/005-june-threshold-mix-collapse.md)
records the run's config (`Device: cuda`, GPU inference server active) and its
164→1440 FPS baseline number, but does not name this note or measure the
server's batch size — see the contrast case below for what that leaves
unreconciled.

## Contrast case: truco-py's 8.8x (unreconciled)

truco-py's own centralized GPU inference server reported the opposite
result: self-play FPS went from ~164 (CPU inference inside subprocesses, one
checkpoint in the pool) to ~1440 after building a daemon-thread server —
"matching threshold-only baseline" —
[`docs/session-2026-06-06.md:80-99`](https://github.com/guidodinello/truco-py/blob/main/docs/session-2026-06-06.md).
truco-py's server (`training/inference_server.py`) uses the same pattern
catan's does: a single `wait(self._server_conns, timeout=self._BATCH_WAIT_S)`
call per loop iteration, `_BATCH_WAIT_S = 0.005`, with no draining of
requests that arrive after the first one is ready. By this note's mechanism,
that predicts a low mean batch size in truco-py too — but truco-py's own
docs never instrument or report a batch size, so this is not confirmed, only
predicted by analogy.

Two candidate explanations for the different outcome, both unmeasured and
stated here only as hypotheses, not conclusions:

- **A much larger Amdahl ceiling.** "Matching threshold-only baseline" at
  1440 fps implies opponent inference was consuming roughly `1 - 1/8.8 ≈
  89%` of wall time before the fix — if true, that alone would explain most
  of the 8.8x even at a mean batch size near 1, since removing an ~89% share
  entirely (by moving it off the CPU-bound subprocess and onto a GPU thread)
  bounds a much higher ceiling than catan's 25.8%. This is inferred from the
  reported FPS numbers, not from a profiler pass — truco-py has no
  equivalent of catan's `rl/profile.py`.
- **Removed CPU thread oversubscription, not GPU batching.** truco-py's
  source has no call to `torch.set_num_threads`, `OMP_NUM_THREADS`, or
  `MKL_NUM_THREADS` anywhere (grepped across the whole repository, excluding
  `.venv`/`.git`) — the failure mode [007](007-inference-thread-oversubscription.md)
  describes (every worker independently sizing its thread pool off the
  host's core count) is a plausible, unconfirmed contributor to the 164 fps
  baseline. Moving inference off the CPU subprocesses and onto a single GPU
  thread would remove that contention regardless of whether the server ever
  batches.

Both catan's negative result and truco-py's positive one are reported
honestly; they are **not yet reconciled** — the missing piece in truco-py is
the same instrumentation catan added (`InferenceServer.stats()`-style batch
size counting), which was never built there.

## Related notes

- [007 — Inference-worker thread oversubscription](007-inference-thread-oversubscription.md)
