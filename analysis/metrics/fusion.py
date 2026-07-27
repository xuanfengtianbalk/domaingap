"""Fusion methods.

  ARF: per-axis alpha + cos_sim gate (retained for reference)
  H:   Alpha-Joint Correction — 2D lookup (total_std × pred) corrects each axis
"""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator


def _alpha_joint_correct(coord_a, gt, total_std, gt_ranges, step, n_pred_bins=15) -> np.ndarray:
    """Per-axis correction using 2D alpha lookup table.

    alpha = (GT - pred) / (pred * total_std)   per (total_std_bin, pred_bin) cell.
    pred_corrected = pred + alpha * pred * total_std.
    Pred bins from config gt_ranges, overflow → outermost bins.
    """
    eps = 1e-3
    corrected = coord_a.copy()
    keys = ["x", "y", "z"]
    for axis_idx in range(3):
        ca = coord_a[:, axis_idx]
        gt_axis = gt[:, axis_idx]
        ts = total_std

        v = np.abs(ca) >= eps
        if v.sum() < 10:
            continue
        cv, gv, tv = ca[v], gt_axis[v], ts[v]
        alpha_v = (gv - cv) / (cv * tv + 1e-12)

        ts_edges = np.percentile(tv, np.linspace(0, 100, 11))
        gr = gt_ranges[keys[axis_idx]]
        inner = np.linspace(gr[0], gr[1], n_pred_bins + 1)
        p_edges = np.concatenate([[-np.inf], inner, [np.inf]])

        for i, (tlo, thi) in enumerate(zip(ts_edges[:-1], ts_edges[1:])):
            m_t = (tv >= tlo) & (tv < thi)
            for j, (plo, phi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
                m_p = (cv >= plo) & (cv < phi)
                m = m_t & m_p
                if m.sum() < 5:
                    continue
                a = alpha_v[m]
                if m.sum() >= 10:
                    lo_a, hi_a = np.percentile(a, [trim_pct * 100, (1 - trim_pct) * 100])
                    ma = a[(a >= lo_a) & (a <= hi_a)].mean()
                else:
                    ma = a.mean()
                # apply correction to ALL pixels in this cell
                # (including those with |pred|<eps — they keep original value)
                m_full = (ts >= tlo) & (ts < thi) & (ca >= plo) & (ca < phi)
                corrected[m_full, axis_idx] = ca[m_full] + ma * ca[m_full] * ts[m_full]

    return corrected


def compute(stats: StatsAccumulator) -> dict:
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"error": "no epi_var_A"}

    epi = stats.epi_var_A.flatten()
    ca, cb, gt = stats.coord_A, stats.coord_B, stats.coord_gt   # (N,3)

    # ── baselines ──
    result = {
        "DER_mae": float(stats.error_A.mean()),
        "gs_mae":  float(stats.error_B.mean()),
        "avg_mae": float(np.linalg.norm((ca + cb) / 2.0 - gt, axis=1).mean()),
        "n":       int(len(epi)),
    }

    # ── per-axis absolute error ──
    ae_a = np.abs(ca - gt)      # (N, 3)
    ae_b = np.abs(cb - gt)
    gain_xyz = ae_a - ae_b       # (N, 3)  >0 → GS better on that axis

    # ── cos_sim per pixel ──
    da = ca - gt
    db = cb - gt
    na = np.linalg.norm(da, axis=1)
    nb = np.linalg.norm(db, axis=1)
    denom = na * nb
    valid = denom > 1e-12
    cos_sim = np.ones_like(na)
    cos_sim[valid] = np.clip((da[valid] * db[valid]).sum(axis=1) / denom[valid], -1.0, 1.0)

    # ── 20-ventile bins ──
    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(np.round(edges, 8))
    n_bins = len(edges) - 1

    mean_gain_x = np.zeros(n_bins)
    mean_gain_y = np.zeros(n_bins)
    mean_gain_z = np.zeros(n_bins)
    mean_cos_sim = np.zeros(n_bins)
    bin_counts  = np.zeros(n_bins, dtype=int)

    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:])):
        m = (epi >= lo) & (epi < hi)
        n = m.sum()
        if n < 3:
            continue
        mean_gain_x[i] = gain_xyz[m, 0].mean()
        mean_gain_y[i] = gain_xyz[m, 1].mean()
        mean_gain_z[i] = gain_xyz[m, 2].mean()
        mean_cos_sim[i] = cos_sim[m].mean()
        bin_counts[i] = n

    gmax_x = max(mean_gain_x.max(), 1e-12)
    gmax_y = max(mean_gain_y.max(), 1e-12)
    gmax_z = max(mean_gain_z.max(), 1e-12)

    alpha_x = np.clip(mean_gain_x / gmax_x, 0.0, 1.0)
    alpha_y = np.clip(mean_gain_y / gmax_y, 0.0, 1.0)
    alpha_z = np.clip(mean_gain_z / gmax_z, 0.0, 1.0)

    # ── apply ARF with consistency gate ──
    fused = ca.copy()                                          # default = DER
    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:])):
        if bin_counts[i] < 3:
            continue
        m = (epi >= lo) & (epi < hi)
        if mean_cos_sim[i] > 0.9:
            fused[m, 0] = ca[m, 0] + alpha_x[i] * (cb[m, 0] - ca[m, 0])
            fused[m, 1] = ca[m, 1] + alpha_y[i] * (cb[m, 1] - ca[m, 1])
            fused[m, 2] = ca[m, 2] + alpha_z[i] * (cb[m, 2] - ca[m, 2])

    e_fused = np.linalg.norm(fused - gt, axis=1)
    result["ARF_mae"]  = float(e_fused.mean())
    result["ARF_rmse"] = float(np.sqrt((e_fused ** 2).mean()))
    result["n_consistent_bins"] = int(sum(mean_cos_sim > 0.9))
    result["n_total_bins"] = n_bins

    # ── alpha table ──
    alpha_rows = []
    for i in range(n_bins):
        if bin_counts[i] < 3:
            continue
        alpha_rows.append({
            "bin":       i,
            "epi_lo":    float(edges[i]),
            "epi_hi":    float(edges[i + 1]),
            "epi_mean":  float(epi[(epi >= edges[i]) & (epi < edges[i + 1])].mean()),
            "n":         int(bin_counts[i]),
            "cos_sim":   float(mean_cos_sim[i]),
            "gated":     bool(mean_cos_sim[i] > 0.9),
            "gain_x":    float(mean_gain_x[i]), "gain_y": float(mean_gain_y[i]), "gain_z": float(mean_gain_z[i]),
            "alpha_x":   float(alpha_x[i]),     "alpha_y": float(alpha_y[i]),     "alpha_z": float(alpha_z[i]),
        })
    result["alpha_table"] = alpha_rows

    # ── per-decile MAE ──
    dec_edges = np.percentile(epi, np.linspace(0, 100, 11))
    dec_centers = [(i * 10 + (i + 1) * 10) / 2 for i in range(10)]

    # ── H: Alpha-Joint Correction ──
    import yaml, os
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    gr = cfg["gt_ranges"]
    step = gr["bin_step"]
    trim_pct = gr.get("trim_pct", 0.1)
    n_pred_bins = gr.get("n_pred_bins", 15)

    if stats.alea_var_A is not None and len(stats.alea_var_A) > 0:
        total_var = np.maximum(stats.epi_var_A.flatten() + stats.alea_var_A.flatten(), 0.0)
        total_std = np.sqrt(total_var)
    else:
        total_std = np.sqrt(np.maximum(stats.epi_var_A.flatten(), 0.0))

    ca_corr = _alpha_joint_correct(ca, gt, total_std, gr, step, n_pred_bins)
    e_corr = np.linalg.norm(ca_corr - gt, axis=1)
    result["H_alpha_correct_mae"]  = float(e_corr.mean())
    result["H_alpha_correct_rmse"] = float(np.sqrt((e_corr ** 2).mean()))

    # ── per-decile (add H) ──
    decile = []
    for d in range(10):
        lo, hi = dec_edges[d], dec_edges[d + 1]
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 10:
            continue
        ci, cj, gm = ca[m], cb[m], gt[m]
        decile.append({
            "pct_center": dec_centers[d],
            "n": int(m.sum()),
            "DER": float(np.linalg.norm(ci - gm, axis=1).mean()),
            "gs":  float(np.linalg.norm(cj - gm, axis=1).mean()),
            "avg": float(np.linalg.norm((ci + cj) / 2.0 - gm, axis=1).mean()),
            "ARF": float(np.linalg.norm(fused[m] - gm, axis=1).mean()),
            "H_alpha_correct": float(np.linalg.norm(ca_corr[m] - gm, axis=1).mean()),
        })
    result["per_decile"] = decile

    return result
