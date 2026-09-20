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

Gaps 1 (unify the vectorized/iterative sampling models), 4 (variance
reduction) and 5 (adaptive stopping) are tracked as issues on this repo, not
addressed here -- see ``docs/shared-ml-package.md`` for links.
"""

from __future__ import annotations

from gamekit.mc.accumulate import Accumulator, MeanAccumulator
from gamekit.mc.intervals import ConfidenceInterval, MCResult, wilson_interval
from gamekit.mc.sample_size import (
    sample_size_clt,
    sample_size_hoeffding,
    two_proportion_sample_size,
)
from gamekit.mc.testing import (
    TwoProportionTest,
    benjamini_hochberg,
    two_proportion_test,
)

__all__ = [
    "Accumulator",
    "ConfidenceInterval",
    "MCResult",
    "MeanAccumulator",
    "TwoProportionTest",
    "benjamini_hochberg",
    "sample_size_clt",
    "sample_size_hoeffding",
    "two_proportion_sample_size",
    "two_proportion_test",
    "wilson_interval",
]
