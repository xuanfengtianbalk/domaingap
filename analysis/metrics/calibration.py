"""Uncertainty calibration: error / gain / winrate vs epi_std profiles.

Purpose: find the epistemic uncertainty threshold τ where Head2 provides
meaningful gains over Head1, so Head2 can supervise Head1 on uncertain pixels.
"""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator

NO_CONDITIONS = True


def _per_axis_gain(stats: StatsAccumulator, mask: np.ndarray, label: str) -> dict:
    ca = stats.coord_A[mask]
    cb = stats.coord_B[mask]
    gt = stats.coord_gt[mask]
    axes = ["x", "y", "z"]
    result = {"_mask_label": label, "n": int(mask.sum())}
    for i, ax in enumerate(axes):
        e_a = np.abs(ca[:, i] - gt[:, i]).mean()
        e_b = np.abs(cb[:, i] - gt[:, i]).mean()
        result[f"mae_{ax}_A"] = float(e_a)
        result[f"mae_{ax}_B"] = float(e_b)
        result[f"gain_{ax}"] = float(e_a - e_b)
    return result


def compute(stats: StatsAccumulator) -> dict:
    """Profile model error / gain / win rate as a function of epistemic std.

    Returns binned data and threshold-sweep table suitable for τ selection.
    """
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"error": "no epi_var_A data", "n_pixels": 0}

    epi = stats.epi_var_A.flatten()
    ea = stats.error_A
    eb = stats.error_B
    n_total = len(epi)

    # ---- epi percentile reference points ----
    epi_pct = {str(p): float(np.percentile(epi, p)) for p in [25, 33, 50, 66, 75, 90, 95, 99]}

    # ---- binned error/gain/winrate vs epi_std ----
    bin_edges = np.percentile(epi, np.linspace(0, 100, 21))
    bin_edges = np.unique(np.round(bin_edges, 8))

    error_profile = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 10:
            continue
        ea_bin = ea[m]
        eb_bin = eb[m]
        delta = ea_bin - eb_bin
        error_profile.append({
            "epi_lo": float(lo),
            "epi_hi": float(hi),
            "epi_mean": float(epi[m].mean()),
            "n": int(m.sum()),
            "mae_A":  float(ea_bin.mean()),
            "mae_B":  float(eb_bin.mean()),
            "rmse_A": float(np.sqrt((ea_bin ** 2).mean())),
            "rmse_B": float(np.sqrt((eb_bin ** 2).mean())),
            "mean_gain":   float(delta.mean()),
            "median_gain": float(np.median(delta)),
            "gain_std":    float(delta.std()),
            "win_rate_A":  float((ea_bin < eb_bin).mean()),
        })

    # ---- threshold sweep ----
    pcts = list(range(50, 100, 5)) + [99]
    τ_values = np.percentile(epi, pcts)

    sweep = []
    for pct, τ in zip(pcts, τ_values):
        m = epi >= τ
        n_above = m.sum()
        if n_above < 100:
            continue
        ea_above = ea[m]
        eb_above = eb[m]
        delta_above = ea_above - eb_above
        sweep.append({
            "τ":                 float(τ),
            "τ_pct":             float(pct),
            "pct_pixels_above":  float(n_above / n_total * 100),
            "n_above":           int(n_above),
            "mae_A_above":       float(ea_above.mean()),
            "mae_B_above":       float(eb_above.mean()),
            "rmse_A_above":      float(np.sqrt((ea_above ** 2).mean())),
            "rmse_B_above":      float(np.sqrt((eb_above ** 2).mean())),
            "mean_gain_above":   float(delta_above.mean()),
            "median_gain_above": float(np.median(delta_above)),
            "win_rate_A_above":  float((ea_above < eb_above).mean()),
        })

    # ---- per-axis gain at p90 (high-uncertainty tail) ----
    τ_p90 = epi_pct["90"]
    m_p90 = epi >= τ_p90
    per_axis_p90 = _per_axis_gain(stats, m_p90, "epi_std >= p90")

    return {
        "n_pixels": int(n_total),
        "epi_pct":  epi_pct,
        "error_profile":   error_profile,
        "threshold_sweep": sweep,
        "per_axis_above_p90": per_axis_p90,
    }
