"""Uncertainty-conditioned metrics (DER / NIG models).

Requires ``epi_var_A`` and / or ``alea_var_A`` in the accumulator.
"""

from __future__ import annotations
import numpy as np
from accumulator import StatsAccumulator


def compute(stats: StatsAccumulator, condition: str = "overall") -> dict:
    """Compute uncertainty-aware statistics.

    Args:
        stats: Accumulator with optional ``epi_var_A``, ``alea_var_A``.
        condition: One of ``"overall"``, ``"low_epi"``, ``"high_epi"``.

    Returns:
        dict with ``mean_error_A``, ``mean_error_B``, ``n``.
    """
    mask = _condition_mask(stats, condition)
    e_a = stats.error_A[mask] if len(stats.error_A) > 0 else np.array([])
    e_b = stats.error_B[mask] if len(stats.error_B) > 0 else np.array([])
    return {
        "condition": condition,
        "mean_error_A": float(e_a.mean()) if len(e_a) > 0 else None,
        "mean_error_B": float(e_b.mean()) if len(e_b) > 0 else None,
        "n": int(len(e_a)),
    }


def _condition_mask(stats: StatsAccumulator, cond: str) -> np.ndarray:
    """Return a boolean mask for the given condition."""
    if cond == "overall":
        return np.ones(len(stats.error_A), dtype=bool)
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return np.ones(len(stats.error_A), dtype=bool)
    epi = stats.epi_var_A.flatten()
    lo, hi = np.percentile(epi, [33, 66])
    if cond == "low_epi":
        return epi < lo
    if cond == "mid_epi":
        return (epi >= lo) & (epi < hi)
    if cond == "high_epi":
        return epi >= hi
    return np.ones(len(stats.error_A), dtype=bool)
