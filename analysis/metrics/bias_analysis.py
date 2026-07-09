"""Bias analysis: statistical validation before modelling.

Four analyses:
  1. bias_correlation  — corr(bias_DER, bias_gs) per decile
  2. bias_magnitude    — E(|bias| | u_epi) per ventile
  3. signed_bias       — E(bias_x/y/z | u_epi) per ventile
  4. bias_angle        — arccos(e_DER · e_GS / |e_DER||e_GS|) per ventile

Output: bias_analysis.json (combined)
"""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator

NO_CONDITIONS = True


def _decile_corr(stats: StatsAccumulator) -> dict:
    """Pearson r between bias_DER and bias_gs, per decile of epi_std."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"error": "no epi_var_A"}

    epi = stats.epi_var_A.flatten()
    diff_a = stats.diff_A   # (N, 3)  coords_A - GT
    diff_b = stats.diff_B   # (N, 3)  coords_B - GT

    bias_norm_a = np.linalg.norm(diff_a, axis=1)
    bias_norm_b = np.linalg.norm(diff_b, axis=1)

    edges = np.percentile(epi, np.linspace(0, 100, 11))
    names = [f"p{lo}-p{hi}" for lo, hi in zip(range(0, 100, 10), range(10, 110, 10))]

    overall = {
        "pearson_r":   float(np.corrcoef(bias_norm_a, bias_norm_b)[0, 1]),
        "axis_x":      float(np.corrcoef(diff_a[:, 0], diff_b[:, 0])[0, 1]),
        "axis_y":      float(np.corrcoef(diff_a[:, 1], diff_b[:, 1])[0, 1]),
        "axis_z":      float(np.corrcoef(diff_a[:, 2], diff_b[:, 2])[0, 1]),
        "n":           int(len(epi)),
    }

    by_decile = {}
    for i, name in enumerate(names):
        lo, hi = edges[i], edges[i + 1]
        m = (epi >= lo) & (epi < hi) if hi < np.inf else (epi >= lo)
        n = m.sum()
        if n < 3:
            continue
        by_decile[name] = {
            "epi_lo":       float(lo),
            "epi_hi":       float(hi) if hi < np.inf else "inf",
            "pearson_r":    float(np.corrcoef(bias_norm_a[m], bias_norm_b[m])[0, 1]),
            "axis_x":       float(np.corrcoef(diff_a[m, 0], diff_b[m, 0])[0, 1]),
            "axis_y":       float(np.corrcoef(diff_a[m, 1], diff_b[m, 1])[0, 1]),
            "axis_z":       float(np.corrcoef(diff_a[m, 2], diff_b[m, 2])[0, 1]),
            "n":            int(n),
        }

    return {"overall": overall, "by_decile": by_decile}


def _ventile_stats(stats: StatsAccumulator, field: str) -> dict:
    """Binned mean of |bias| or signed bias_x/y/z, 20 bins of epi_std."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"error": "no epi_var_A"}

    epi = stats.epi_var_A.flatten()
    diff_a = stats.diff_A
    diff_b = stats.diff_B

    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(np.round(edges, 8))

    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (epi >= lo) & (epi < hi)
        n = m.sum()
        if n < 10:
            continue

        if field == "magnitude":
            entry = {
                "epi_lo": float(lo), "epi_hi": float(hi),
                "epi_mean": float(epi[m].mean()), "n": int(n),
                "|bias|_DER": float(np.linalg.norm(diff_a[m], axis=1).mean()),
                "|bias|_gs":  float(np.linalg.norm(diff_b[m], axis=1).mean()),
            }
        else:  # signed
            entry = {
                "epi_lo": float(lo), "epi_hi": float(hi),
                "epi_mean": float(epi[m].mean()), "n": int(n),
                "bias_x_DER": float(diff_a[m, 0].mean()),
                "bias_y_DER": float(diff_a[m, 1].mean()),
                "bias_z_DER": float(diff_a[m, 2].mean()),
                "bias_x_gs":  float(diff_b[m, 0].mean()),
                "bias_y_gs":  float(diff_b[m, 1].mean()),
                "bias_z_gs":  float(diff_b[m, 2].mean()),
            }
        bins.append(entry)

    return {"bins": bins, "n_total": int(len(epi))}


def _angle_stats(stats: StatsAccumulator) -> dict:
    """arccos(e_DER · e_GS / |e_DER||e_GS|) per ventile — angle between error vectors."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"error": "no epi_var_A"}

    epi = stats.epi_var_A.flatten()
    diff_a = stats.diff_A
    diff_b = stats.diff_B

    norm_a = np.linalg.norm(diff_a, axis=1)
    norm_b = np.linalg.norm(diff_b, axis=1)
    dot = (diff_a * diff_b).sum(axis=1)
    denom = norm_a * norm_b

    valid = (norm_a > 1e-12) & (norm_b > 1e-12)
    cos_sim = np.full_like(dot, np.nan)
    cos_sim[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    angle_deg = np.rad2deg(np.arccos(np.clip(cos_sim, -1.0, 1.0)))

    overall_valid = valid
    overall = {
        "mean_angle_deg": float(np.nanmean(angle_deg[overall_valid])),
        "mean_cos_sim":   float(np.nanmean(cos_sim[overall_valid])),
        "n_valid":        int(overall_valid.sum()),
        "n_total":        int(len(epi)),
    }

    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(np.round(edges, 8))

    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (epi >= lo) & (epi < hi)
        n = m.sum()
        if n < 10:
            continue
        vm = m & valid
        nv = vm.sum()
        if nv < 3:
            continue
        bins.append({
            "epi_lo":         float(lo),
            "epi_hi":         float(hi),
            "epi_mean":       float(epi[m].mean()),
            "n":              int(n),
            "n_valid":        int(nv),
            "mean_angle_deg": float(np.mean(angle_deg[vm])),
            "std_angle_deg":  float(np.std(angle_deg[vm])),
            "mean_cos_sim":   float(np.mean(cos_sim[vm])),
        })

    return {"overall": overall, "bins": bins}


def compute(stats: StatsAccumulator) -> dict:
    return {
        "bias_correlation":  _decile_corr(stats),
        "bias_magnitude":    _ventile_stats(stats, "magnitude"),
        "signed_bias":       _ventile_stats(stats, "signed"),
        "bias_angle":        _angle_stats(stats),
    }
