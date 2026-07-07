"""Pearson & Spearman correlation between per-pixel errors of two models."""

from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute(stats) -> dict:
    """Compute correlation metrics.

    Args:
        stats: ``StatsAccumulator`` with ``error_A`` and ``error_B`` as
            numpy arrays.

    Returns:
        dict with keys ``pearson_r``, ``pearson_p``, ``spearman_r``,
        ``spearman_p``, ``n``.
    """
    e_a = stats.error_A
    e_b = stats.error_B
    n = len(e_a)
    if n < 3:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None, "n": n}
    pr, pp = pearsonr(e_a, e_b)
    sr, sp = spearmanr(e_a, e_b)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp), "n": int(n)}
