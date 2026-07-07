"""Error gain Δ = error_A − error_B.  Positive → Head2 is better."""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator


def compute(stats: StatsAccumulator) -> dict:
    """Per-pixel error difference between two models.

    Args:
        stats: Accumulator with ``error_A`` and ``error_B``.

    Returns:
        dict with mean/median gain, std, and fraction where Head2 wins.
    """
    e_a = stats.error_A
    e_b = stats.error_B
    n = len(e_a)
    if n == 0:
        return {"mean_gain": None, "median_gain": None, "n": 0}

    delta = e_a - e_b         # positive → Head2 is better
    return {
        "mean_gain":            float(delta.mean()),
        "median_gain":          float(np.median(delta)),
        "gain_std":             float(delta.std()),
        "pct_head2_wins":       float((delta > 0).mean()),
        "mean_gain_head2_wins": float(delta[delta > 0].mean()) if (delta > 0).any() else None,
        "n": int(n),
    }
