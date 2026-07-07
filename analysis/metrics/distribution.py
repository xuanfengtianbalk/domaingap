"""Bin distribution analysis (gs models).

Requires ``bin_probs_A`` and ``entropy_A`` in the accumulator.
"""

from __future__ import annotations
import numpy as np
from accumulator import StatsAccumulator


def compute(stats: StatsAccumulator) -> dict:
    """Compute distribution metrics: entropy-error correlation, peak/top3.

    Returns:
        dict with ``entropy_error_corr``, ``peak_mean``, ``top3_mean``.
    """
    result: dict = {}
    if stats.entropy_A is not None and len(stats.entropy_A) > 0:
        ent = stats.entropy_A
        err = stats.error_A if len(stats.error_A) > 0 else stats.error_B
        if len(err) == len(ent) and len(ent) > 2:
            result["entropy_error_corr"] = float(np.corrcoef(ent, err)[0, 1])
    if stats.bin_probs_A is not None and len(stats.bin_probs_A) > 0:
        sorted_probs = -np.sort(-stats.bin_probs_A, axis=-1)[:, :3]
        result["peak_mean"] = float(sorted_probs[:, 0].mean())
        result["top3_mean"] = float(sorted_probs.sum(axis=1).mean())
    result["n_pixels"] = int(len(stats.error_A))
    return result
