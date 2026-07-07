"""MAE / RMSE for each model, designed to work with ConditionAnalyzer."""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator


def compute(stats: StatsAccumulator) -> dict:
    """Per-model MAE and RMSE.

    Args:
        stats: Accumulator with ``error_A`` and ``error_B`` as 1-D arrays.

    Returns:
        dict with ``mae_A``, ``rmse_A``, ``mae_B``, ``rmse_B``, ``n``.
    """
    e_a = stats.error_A
    e_b = stats.error_B
    n = len(e_a)
    if n == 0:
        return {"mae_A": None, "rmse_A": None, "mae_B": None, "rmse_B": None, "n": 0}
    return {
        "mae_A":  float(e_a.mean()),
        "rmse_A": float(np.sqrt((e_a ** 2).mean())),
        "mae_B":  float(e_b.mean()),
        "rmse_B": float(np.sqrt((e_b ** 2).mean())),
        "n": int(n),
    }
