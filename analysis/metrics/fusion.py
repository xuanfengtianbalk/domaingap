"""Gated fusion: low-epi → Head1, mid → average, high → Head2."""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator


def compute(stats: StatsAccumulator) -> dict:
    """Simple uncertainty-gated fusion and constant-weight baselines.

    Uses Head1's epistemic uncertainty to select:
      - low_epi (< p33)           → Head1 only
      - mid_epi ([p33, p66))      → equal-weight average
      - high_epi (>= p66)         → Head2 only

    Also reports constant-weight fusion baselines for comparison.

    Args:
        stats: Accumulator with ``epi_var_A``, ``coord_A``, ``coord_B``,
            ``coord_gt``, ``error_A``, ``error_B``.

    Returns:
        dict with MAE / RMSE for gated fusion and constant-weight variants.
    """
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"gated_mae": None, "gated_rmse": None}

    epi = stats.epi_var_A.flatten()
    lo, hi = np.percentile(epi, [33, 66])

    coord_a = stats.coord_A  # (N, 3)
    coord_b = stats.coord_B
    gt = stats.coord_gt

    # ---- gated fusion ----
    fused = np.where(
        epi[:, None] < lo, coord_a,
        np.where(epi[:, None] < hi, (coord_a + coord_b) / 2.0, coord_b),
    )
    fe = np.linalg.norm(fused - gt, axis=1)

    # ---- baselines ----
    avg = (coord_a + coord_b) / 2.0
    avg_e = np.linalg.norm(avg - gt, axis=1)

    result = {
        "gated_mae":       float(fe.mean()),
        "gated_rmse":      float(np.sqrt((fe ** 2).mean())),
        "avg_mae":         float(avg_e.mean()),
        "avg_rmse":        float(np.sqrt((avg_e ** 2).mean())),
        "head1_mae":       float(stats.error_A.mean()),
        "head1_rmse":      float(np.sqrt((stats.error_A ** 2).mean())),
        "head2_mae":       float(stats.error_B.mean()),
        "head2_rmse":      float(np.sqrt((stats.error_B ** 2).mean())),
        "n":               int(len(stats.error_A)),
    }
    return result
