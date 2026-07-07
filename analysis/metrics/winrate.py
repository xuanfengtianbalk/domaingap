"""Win rate: fraction of pixels where model A error < model B error."""

from __future__ import annotations
import numpy as np


def compute(stats) -> dict:
    """Compute win rate and per-bin win rates.

    Args:
        stats: ``StatsAccumulator`` with ``error_A`` and ``error_B``.

    Returns:
        dict with ``win_rate_A``, ``n_pixels``, ``per_bin``.
    """
    e_a = stats.error_A
    e_b = stats.error_B
    n = len(e_a)
    if n == 0:
        return {"win_rate_A": None, "n_pixels": 0, "per_bin": {}}

    win_rate = float((e_a < e_b).mean())

    diff = np.abs(e_a - e_b)
    thresholds = np.percentile(diff[diff > 0], [25, 50, 75, 90, 95]) if (diff > 0).any() else []

    per_bin = {}
    all_thresholds = [0.0] + list(thresholds) + [float("inf")]
    for lo, hi in zip(all_thresholds[:-1], all_thresholds[1:]):
        mask = (diff >= lo) & (diff < hi)
        if mask.sum() > 0:
            per_bin[f"[{lo:.4f}, {hi:.4f})"] = {
                "win_rate_A": float((e_a[mask] < e_b[mask]).mean()),
                "n": int(mask.sum()),
            }

    return {"win_rate_A": win_rate, "n_pixels": int(n), "per_bin": per_bin}
