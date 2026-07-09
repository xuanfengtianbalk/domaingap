"""Per-axis bias analysis — x, y, z independently, 100 percentile bins.

  1. sign_agreement     — fraction of pixels where bias_DER and bias_gs share sign
  2. bias_magnitude     — mean, var, p25/p50/p75 of |bias| per bin
  3. disagreement_ratio — |DER-gs|/|bias_DER|, |DER-gs|/|bias_gs| per bin
  4. gt_conditioned     — mean/var bias per GT coordinate range
"""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator

NO_CONDITIONS = True


def _axis_analysis(epi, diff_a, diff_b, gt, axis_idx: int) -> dict:
    da = diff_a[:, axis_idx]    # bias_DER per axis  (N,)
    db = diff_b[:, axis_idx]    # bias_gs  per axis
    gt_axis = gt[:, axis_idx]   # GT per axis

    # ── 100 percentile bins ──
    edges = np.percentile(epi, np.linspace(0, 100, 101))
    edges = np.unique(np.round(edges, 10))
    n_bins = len(edges) - 1
    pct_centers = [(i + 0.5) * (100.0 / n_bins) for i in range(n_bins)]

    # ── 1. sign agreement ──
    sign_agree = []
    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:])):
        m = (epi >= lo) & (epi < hi)
        n = m.sum()
        if n < 3:
            sign_agree.append({"pct": pct_centers[i], "frac": None, "n": 0})
            continue
        same_sign = ((da[m] > 0) & (db[m] > 0)) | ((da[m] < 0) & (db[m] < 0))
        sign_agree.append({
            "pct":   pct_centers[i],
            "frac":  float(same_sign.mean()),
            "n":     int(n),
        })

    # ── 2. bias magnitude distribution ──
    bias_mag = []
    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:])):
        m = (epi >= lo) & (epi < hi)
        n = m.sum()
        if n < 3:
            bias_mag.append({"pct": pct_centers[i], "n": 0})
            continue
        ada, adb = np.abs(da[m]), np.abs(db[m])
        bias_mag.append({
            "pct":       pct_centers[i],
            "n":         int(n),
            "mean_DER":  float(ada.mean()),  "var_DER": float(ada.var()),
            "p25_DER":   float(np.percentile(ada, 25)),
            "p50_DER":   float(np.percentile(ada, 50)),
            "p75_DER":   float(np.percentile(ada, 75)),
            "mean_gs":   float(adb.mean()),  "var_gs":  float(adb.var()),
            "p25_gs":    float(np.percentile(adb, 25)),
            "p50_gs":    float(np.percentile(adb, 50)),
            "p75_gs":    float(np.percentile(adb, 75)),
        })

    # ── 3. disagreement ratio ──
    eps = 1e-12
    disagree = []
    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:])):
        m = (epi >= lo) & (epi < hi)
        n = m.sum()
        if n < 3:
            disagree.append({"pct": pct_centers[i], "n": 0})
            continue
        ratio_der = np.abs(da[m] - db[m]) / (np.abs(da[m]) + eps)
        ratio_gs  = np.abs(da[m] - db[m]) / (np.abs(db[m]) + eps)
        disagree.append({
            "pct":            pct_centers[i],
            "n":              int(n),
            "mean_vs_DER":    float(ratio_der.mean()),
            "var_vs_DER":     float(ratio_der.var()),
            "p25_vs_DER":     float(np.percentile(ratio_der, 25)),
            "p50_vs_DER":     float(np.percentile(ratio_der, 50)),
            "p75_vs_DER":     float(np.percentile(ratio_der, 75)),
            "mean_vs_gs":     float(ratio_gs.mean()),
            "var_vs_gs":      float(ratio_gs.var()),
            "p25_vs_gs":      float(np.percentile(ratio_gs, 25)),
            "p50_vs_gs":      float(np.percentile(ratio_gs, 50)),
            "p75_vs_gs":      float(np.percentile(ratio_gs, 75)),
        })

    # ── 4. GT-conditioned bias ──
    gt_min, gt_max = gt_axis.min(), gt_axis.max()
    gt_edges = np.linspace(gt_min, gt_max, 21)   # 20 equal-width bins
    gt_conditioned = []
    for lo, hi in zip(gt_edges[:-1], gt_edges[1:]):
        m = (gt_axis >= lo) & (gt_axis < hi)
        n = m.sum()
        if n < 3:
            continue
        gt_conditioned.append({
            "gt_lo":          float(lo),
            "gt_hi":          float(hi),
            "n":              int(n),
            "mean_bias_DER":  float(da[m].mean()),
            "var_bias_DER":   float(da[m].var()),
            "mean_bias_gs":   float(db[m].mean()),
            "var_bias_gs":    float(db[m].var()),
        })

    return {
        "sign_agreement":     sign_agree,
        "bias_magnitude":     bias_mag,
        "disagreement_ratio": disagree,
        "gt_conditioned":     gt_conditioned,
    }


def compute(stats: StatsAccumulator) -> dict:
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"error": "no epi_var_A"}

    epi = stats.epi_var_A.flatten()
    da = stats.diff_A    # (N, 3)  DER - GT
    db = stats.diff_B    # (N, 3)  gs  - GT
    gt = stats.coord_gt

    return {
        "x": _axis_analysis(epi, da, db, gt, 0),
        "y": _axis_analysis(epi, da, db, gt, 1),
        "z": _axis_analysis(epi, da, db, gt, 2),
    }
