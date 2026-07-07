"""Residual analysis (Fusion / two-head comparison).

Computes residual = coord_A - coord_B and its relationship with uncertainty.
"""
from __future__ import annotations
import numpy as np
from accumulator import StatsAccumulator


def compute(stats: StatsAccumulator) -> dict:
    """Compute residual between two model predictions.

    Returns:
        dict with ``residual_mean``, ``residual_std``,
        ``residual_epi_corr`` (if epi_var available).
    """
    result: dict = {}
    if len(stats.coord_A) > 0 and len(stats.coord_B) > 0:
        residual = stats.coord_A - stats.coord_B  # (N, 3)
        res_norm = np.linalg.norm(residual, axis=1)
        result["residual_mean"] = float(res_norm.mean())
        result["residual_std"] = float(res_norm.std())
    if stats.epi_var_A is not None and len(stats.epi_var_A) > 0:
        epi = stats.epi_var_A.flatten()
        if len(epi) == len(res_norm) and len(epi) > 2:
            result["residual_epi_corr"] = float(np.corrcoef(epi, res_norm)[0, 1])
    result["n_pixels"] = int(len(stats.error_A))
    return result
