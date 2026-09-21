# 007 — Inference-worker thread oversubscription

**Status:** validated
**Last touched:** 2026-09-20

## Hypothesis

Loading a torch checkpoint inside a parallel worker process without
pinning its intra-op thread count causes every worker to independently
size its BLAS/torch thread pool off the *host's* core count rather than
its own share of it — quietly serializing an "embarrassingly parallel"
benchmark through OS scheduler contention, with no error and no visible
symptom except everything crawling.

## Why we believe it

- `torch.set_num_threads` "Sets the number of threads used for intraop
  parallelism on CPU", and "must be called before running eager, JIT or
  autograd code" — [PyTorch, `torch.set_num_threads`
  docs](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html)
  — takeaway: this has to be set once per process, early, and torch will
  not do it for you based on how many other processes exist.
- catan measured the failure mode directly: **load average ~210 on 20
  cores** while running a benchmark, 16 worker processes each independently
  spinning up ~33 threads — no external source; observed in catan
  `agents/rl_agent.py:36-58` (PR #18).
- Fixing it with `torch.set_num_threads(1)` once per worker process,
  before the first forward pass, took an n=200 benchmark from **>300s,
  timed out** to **7.81s** — and, crucially, produced **bit-identical win
  counts (163/200)** at `workers=1` vs `workers=8` with the fix applied,
  confirming the fix only removes wasted scheduling overhead and doesn't
  change the computation — no external source; observed in catan PR #18's
  body (the timing numbers are not reproduced in any committed file,
  only the fix itself is, at `agents/rl_agent.py:61`).
- `OMP_NUM_THREADS` was considered and rejected for this fix: it has to be
  set before the process starts, which isn't available at the point the
  checkpoint loader runs, and torch's own setter is unconditional besides
  — no external source; observed in catan `agents/rl_agent.py:36-58`.

## How to test

- **Metric:** wall-clock time for a fixed-`n` parallel benchmark, plus a
  win-count identity check between different worker counts (same seeds,
  same checkpoint) to confirm the fix is scheduling-only.
- **Gate:** no timeout, and win counts match exactly across worker counts.
- **Cost estimate:** near zero — one line (`torch.set_num_threads(1)`)
  placed once per worker process, before the first forward pass.

## Result

catan PR #18: benchmark went from **>300s, timed out** (16 workers × ~33
threads on 20 cores) to **7.81s**, with bit-identical win counts (163/200)
between `workers=1` and `workers=8`. The authoritative n=4000 gate run
(see [003](003-discount-horizon.md)) completed in 107.8s with the fix; the
unfixed 16-worker run it replaced produced zero output in 5+ minutes
before being killed.

## Related notes

- [003 — Discount horizon vs episode length](003-discount-horizon.md)
