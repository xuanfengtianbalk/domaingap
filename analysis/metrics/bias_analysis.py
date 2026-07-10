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


def _axis_analysis(epi, diff_a, diff_b, coord_a, coord_b, gt, axis_idx: int) -> dict:
    da = diff_a[:, axis_idx]    # bias_DER per axis  (N,)
    db = diff_b[:, axis_idx]    # bias_gs  per axis
    ca = coord_a[:, axis_idx]   # pred_DER per axis
    cb = coord_b[:, axis_idx]   # pred_gs  per axis
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
    gt_edges = np.percentile(gt_axis, np.linspace(0, 100, 21))
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
            "mean_GT":        float(gt_axis[m].mean()),
            "mean_bias_DER":  float(da[m].mean()),
            "mean_pred_DER":  float(ca[m].mean()),
            "p25_pred_DER":   float(np.percentile(ca[m], 25)),
            "p50_pred_DER":   float(np.percentile(ca[m], 50)),
            "p75_pred_DER":   float(np.percentile(ca[m], 75)),
            "var_bias_DER":   float(da[m].var()),
            "mean_bias_gs":   float(db[m].mean()),
            "mean_pred_gs":   float(cb[m].mean()),
            "p25_pred_gs":    float(np.percentile(cb[m], 25)),
            "p50_pred_gs":    float(np.percentile(cb[m], 50)),
            "p75_pred_gs":    float(np.percentile(cb[m], 75)),
            "var_bias_gs":    float(db[m].var()),
        })

    # ── 5. CT profile: epi_std vs bias, per GT bin ──
    ct_profile = []
    for lo, hi in zip(gt_edges[:-1], gt_edges[1:]):
        m_gt = (gt_axis >= lo) & (gt_axis < hi)
        n = m_gt.sum()
        if n < 30:
            continue
        epi_sub = epi[m_gt]
        ca_sub, cb_sub = ca[m_gt], cb[m_gt]
        gt_sub = gt_axis[m_gt]

        epi_edges = np.percentile(epi_sub, np.linspace(0, 100, 11))
        epi_bins = []
        for elo, ehi in zip(epi_edges[:-1], epi_edges[1:]):
            me = (epi_sub >= elo) & (epi_sub < ehi)
            if me.sum() < 5:
                continue
            epi_bins.append({
                "epi_lo":          float(elo),
                "epi_hi":          float(ehi),
                "mean_epi_std":    float(epi_sub[me].mean()),
                "n":               int(me.sum()),
                "mean_GT":         float(gt_sub[me].mean()),
                "mean_bias_DER":   float((ca_sub[me] - gt_sub[me]).mean()),
                "med_bias_DER":    float(np.percentile(ca_sub[me] - gt_sub[me], 50)),
                "mean_bias_gs":    float((cb_sub[me] - gt_sub[me]).mean()),
                "med_bias_gs":     float(np.percentile(cb_sub[me] - gt_sub[me], 50)),
            })
        ct_profile.append({
            "gt_lo":    float(lo),
            "gt_hi":    float(hi),
            "mean_GT":  float(gt_sub.mean()),
            "epi_bins": epi_bins,
        })

    return {
        "sign_agreement":     sign_agree,
        "bias_magnitude":     bias_mag,
        "disagreement_ratio": disagree,
        "gt_conditioned":     gt_conditioned,
        "ct_profile":         ct_profile,
    }


def compute(stats: StatsAccumulator) -> dict:
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"error": "no epi_var_A"}

    epi = stats.epi_var_A.flatten()
    da = stats.diff_A    # (N, 3)  DER - GT
    db = stats.diff_B    # (N, 3)  gs  - GT
    ca = stats.coord_A   # (N, 3)  DER predictions
    cb = stats.coord_B   # (N, 3)  gs  predictions
    gt = stats.coord_gt

    return {
        "x": _axis_analysis(epi, da, db, ca, cb, gt, 0),
        "y": _axis_analysis(epi, da, db, ca, cb, gt, 1),
        "z": _axis_analysis(epi, da, db, ca, cb, gt, 2),
    }
