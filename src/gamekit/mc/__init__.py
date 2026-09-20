"""Generic Monte Carlo statistics: confidence intervals, sample-size
calculations, two-proportion testing, and streaming accumulators.

Stdlib-only. Extracted from catan's ``experiments/mcstats.py``, which itself
mirrored (only partially -- see ``docs/shared-ml-package.md``) the naming used
by ``mmo-utils`` (``github.com/guidodinello/mmo-utils``), a coursework MC
library. This package resolves two of that library's documented ``DESIGN.md``
gaps directly:

- **Gap 2** (``Accumulator.update`` locked to ``float``) -- see
  ``gamekit.mc.accumulate.Accumulator[V, S, R]``.
- **Gap 3** (no confidence intervals for custom accumulators) -- any
  accumulator that finalizes into ``MCResult`` gets
  ``MCResult.confidence_interval`` for free.
- **Gap 1** (unify the vectorized/iterative sampling models) -- see
  ``gamekit.mc.sample``: one ``Sampler -> Evaluator -> Accumulator`` fold,
  with ``monte_carlo`` a convenience over ``monte_carlo_reduce``.
- **Gap 4** (variance reduction) -- see ``gamekit.mc.variance``: antithetic
  variates as a sampler wrapper, control variates as an evaluator wrapper.
- **Gap 5** (adaptive stopping) -- see ``gamekit.mc.stopping``:
  ``monte_carlo_until`` runs the gap-1 fold in chunks until a target
  confidence-interval half-width is reached or a hard cap is hit.
"""

from __future__ import annotations

from gamekit.mc.accumulate import Accumulator, MeanAccumulator
from gamekit.mc.intervals import ConfidenceInterval, MCResult, wilson_interval
from gamekit.mc.sample import (
    Evaluator,
    Sampler,
    WelfordAccumulator,
    WelfordState,
    monte_carlo,
    monte_carlo_reduce,
    scalar_sampler,
)
from gamekit.mc.sample_size import (
    sample_size_clt,
    sample_size_hoeffding,
    two_proportion_sample_size,
)
from gamekit.mc.stopping import AdaptiveResult, monte_carlo_until
from gamekit.mc.testing import (
    TwoProportionTest,
    benjamini_hochberg,
    two_proportion_test,
)
from gamekit.mc.variance import antithetic, control_beta, control_variate, paired_mean

__all__ = [
    "Accumulator",
    "AdaptiveResult",
    "ConfidenceInterval",
    "Evaluator",
    "MCResult",
    "MeanAccumulator",
    "Sampler",
    "TwoProportionTest",
    "WelfordAccumulator",
    "WelfordState",
    "antithetic",
    "benjamini_hochberg",
    "control_beta",
    "control_variate",
    "monte_carlo",
    "monte_carlo_reduce",
    "monte_carlo_until",
    "paired_mean",
    "sample_size_clt",
    "sample_size_hoeffding",
    "scalar_sampler",
    "two_proportion_sample_size",
    "two_proportion_test",
    "wilson_interval",
]
