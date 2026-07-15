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


def _axis_analysis(epi, total_std, diff_a, diff_b, coord_a, coord_b, gt, gt_range, step, axis_idx: int) -> dict:
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

    # ── 5. CT profile: total_std vs bias, fixed-step GT bins ──
    ct_edges = np.arange(gt_range[0], gt_range[1] + step * 0.5, step)
    ct_profile = []
    for lo, hi in zip(ct_edges[:-1], ct_edges[1:]):
        m_gt = (gt_axis >= lo) & (gt_axis < hi)
        n = m_gt.sum()
        if n < 30:
            continue
        ts_sub = total_std[m_gt]
        ca_sub, cb_sub = ca[m_gt], cb[m_gt]
        gt_sub = gt_axis[m_gt]

        ts_edges = np.percentile(ts_sub, np.linspace(0, 100, 11))
        epi_bins = []
        for elo, ehi in zip(ts_edges[:-1], ts_edges[1:]):
            me = (ts_sub >= elo) & (ts_sub < ehi)
            if me.sum() < 5:
                continue
            epi_bins.append({
                "total_std_lo":     float(elo),
                "total_std_hi":     float(ehi),
                "mean_total_std":   float(ts_sub[me].mean()),
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


def _alpha_joint_profile(coord_a, coord_b, gt, total_std, axis_idx: int, pred_step: str = "0.05", n_ts_bins: int = 10) -> dict:
    """2D bin: total_std (rows, n_ts_bins bins) × pred (cols, ~step bins), compute alpha."""
    ca = coord_a[:, axis_idx]
    gt_axis = gt[:, axis_idx]
    eps = 1e-3

    valid = np.abs(ca) >= eps
    ca_v, gt_v, ts_v = ca[valid], gt_axis[valid], total_std[valid]
    alpha_v = (gt_v - ca_v) / (ca_v * ts_v + 1e-12)

    ts_edges = np.percentile(ts_v, np.linspace(0, 100, n_ts_bins + 1))
    # pred: ~step bins from min to max
    p_min, p_max = ca_v.min(), ca_v.max()
    p_edges = np.arange(p_min, p_max + float(pred_step) * 0.5, float(pred_step))
    p_edges = np.unique(np.round(p_edges, 8))

    grid = []
    for i, (tlo, thi) in enumerate(zip(ts_edges[:-1], ts_edges[1:])):
        m_t = (ts_v >= tlo) & (ts_v < thi)
        for j, (plo, phi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
            m_p = (ca_v >= plo) & (ca_v < phi)
            m = m_t & m_p
            n = m.sum()
            if n < 5:
                continue
            a = alpha_v[m]
            grid.append({
                "total_std_bin": i,   "total_std_lo": float(tlo),   "total_std_hi": float(thi),
                "pred_bin":      j,   "pred_lo":      float(plo),   "pred_hi":      float(phi),
                "n":             int(n),
                "mean_alpha":    float(a.mean()),    "std_alpha":    float(a.std()),
                "mean_GT":       float(gt_v[m].mean()),
                "mean_pred":     float(ca_v[m].mean()),
                "mean_total_std": float(ts_v[m].mean()),
            })

    return {"grid": grid, "total_std_edges": [float(e) for e in ts_edges], "pred_edges": [float(e) for e in p_edges]}


def compute(stats: StatsAccumulator) -> dict:
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"error": "no epi_var_A"}

    import yaml, os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    gr = cfg["gt_ranges"]
    step = gr["bin_step"]
    n_ts_bins = gr.get("n_total_std_bins", 10)

    epi = stats.epi_var_A.flatten()
    if stats.alea_var_A is not None and len(stats.alea_var_A) > 0:
        total_var = np.maximum(stats.epi_var_A.flatten() + stats.alea_var_A.flatten(), 0.0)
        total_std = np.sqrt(total_var)
    else:
        total_std = np.sqrt(np.maximum(stats.epi_var_A.flatten(), 0.0))
    da = stats.diff_A
    db = stats.diff_B
    ca = stats.coord_A
    cb = stats.coord_B
    gt = stats.coord_gt

    return {
        "x": _axis_analysis(epi, total_std, da, db, ca, cb, gt, gr["x"], step, 0),
        "y": _axis_analysis(epi, total_std, da, db, ca, cb, gt, gr["y"], step, 1),
        "z": _axis_analysis(epi, total_std, da, db, ca, cb, gt, gr["z"], step, 2),
        "alpha_joint": {
            "x": _alpha_joint_profile(ca, cb, gt, total_std, 0, "0.05", n_ts_bins),
            "y": _alpha_joint_profile(ca, cb, gt, total_std, 1, "0.05", n_ts_bins),
            "z": _alpha_joint_profile(ca, cb, gt, total_std, 2, "0.05", n_ts_bins),
        },
    }


def write_csv(result: dict, save_path: str):
    """Write alpha_joint_profile grid to CSV."""
    import csv
    rows = []
    for ax in ['x', 'y', 'z']:
        for cell in result['alpha_joint'][ax]['grid']:
            cell['axis'] = ax
            rows.append(cell)
    if not rows:
        return
    with open(save_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
