"""Per-axis (X / Y / Z) MAE for both models."""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator


def compute(stats: StatsAccumulator) -> dict:
    """MAE for each coordinate axis.

    Args:
        stats: Accumulator with ``coord_A``, ``coord_B``, ``coord_gt``
            as ``(N, 3)`` arrays.

    Returns:
        dict with ``mae_{axis}_{A|B}`` and ``gain_{axis}``.
    """
    da = stats.coord_A - stats.coord_gt      # (N, 3)
    db = stats.coord_B - stats.coord_gt
    result: dict = {}
    for axis_name, idx in [("x", 0), ("y", 1), ("z", 2)]:
        mae_a = float(np.abs(da[:, idx]).mean())
        mae_b = float(np.abs(db[:, idx]).mean())
        result[f"mae_{axis_name}_A"] = mae_a
        result[f"mae_{axis_name}_B"] = mae_b
        result[f"gain_{axis_name}"] = mae_a - mae_b
    result["n"] = int(len(stats.error_A))
    return result
