"""Fusion grounded in bias-analysis calibration tables.

Steps:
  1. Table: bias_strength vs std      → E(|bias_DER|), E(|bias_gs|) per ventile
  2. Table: direction_consistency vs std → E(cos_sim) per ventile
  3. Consistency-gated debiasing: only correct when cos_sim > threshold
  4. Per-decile verification: MAE before/after correction

Methods:
  E: consistency-gated — cos_sim[bin] > thr → debias + avg, else → per-axis
  F: per-axis baseline  — all pixels: (gs_x, gs_y, DER_z)
  G: full-debias         — all pixels: (DER_db + gs_db) / 2  (upper bound)
"""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator


def _mae_rmse(fused, gt):
    e = np.linalg.norm(fused - gt, axis=1)
    return float(e.mean()), float(np.sqrt((e ** 2).mean()))


def _per_axis_fuse(ca, cb):
    fused = cb.copy()
    fused[:, 2] = ca[:, 2]
    return fused


def _build_ventile_tables(epi, coord_a, coord_b, gt):
    """Build 20-ventile bias-strength and direction-consistency tables."""
    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(np.round(edges, 8))

    strength = []
    consistency = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (epi >= lo) & (epi < hi)
        n = m.sum()
        if n < 3:
            strength.append(None)
            consistency.append(None)
            continue

        ca, cb, gm = coord_a[m], coord_b[m], gt[m]
        da, db = ca - gm, cb - gm
        na = np.linalg.norm(da, axis=1)
        nb = np.linalg.norm(db, axis=1)
        denom = na * nb
        valid = denom > 1e-12
        cos_s = np.ones(n)
        cos_s[valid] = np.clip((da[valid] * db[valid]).sum(axis=1) / denom[valid], -1.0, 1.0)

        strength.append({
            "epi_lo": float(lo), "epi_hi": float(hi),
            "epi_mean": float(epi[m].mean()),
            "n": int(n),
            "|bias|_DER": float(na.mean()),
            "|bias|_gs":  float(nb.mean()),
            "bias_x_DER": float(da[:, 0].mean()), "bias_y_DER": float(da[:, 1].mean()), "bias_z_DER": float(da[:, 2].mean()),
            "bias_x_gs":  float(db[:, 0].mean()), "bias_y_gs":  float(db[:, 1].mean()), "bias_z_gs":  float(db[:, 2].mean()),
        })
        consistency.append({
            "epi_lo": float(lo), "epi_hi": float(hi),
            "epi_mean": float(epi[m].mean()),
            "cos_sim_mean": float(cos_s.mean()),
            "n": int(n),
        })

    return edges, strength, consistency


def _consistency_gated(coord_a, coord_b, gt, epi, edges, consistency, thr=0.9):
    """E: consistency-gated debiasing."""
    da_db = coord_a.copy()
    db_db = coord_b.copy()

    n_bins = len(edges) - 1
    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:n_bins+1])):
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 3:
            continue
        da_db[m] -= (coord_a[m] - gt[m]).mean(axis=0)
        db_db[m] -= (coord_b[m] - gt[m]).mean(axis=0)

    # Gate: where cos_sim > thr → avg debiased, else → per-axis
    fused = _per_axis_fuse(coord_a, coord_b)   # default: per-axis everywhere
    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:n_bins+1])):
        cs = consistency[i]
        if cs is None:
            continue
        m = (epi >= lo) & (epi < hi)
        if cs["cos_sim_mean"] > thr:
            fused[m] = (da_db[m] + db_db[m]) / 2.0

    mae, rmse = _mae_rmse(fused, gt)
    n_cons = sum(1 for cs in consistency if cs and cs["cos_sim_mean"] > thr)
    return {"mae": mae, "rmse": rmse, "consistency_thr": thr,
            "n_consistent_bins": n_cons, "n_total_bins": len(consistency)}


def _per_decile_verification(coord_a, coord_b, gt, epi, edges, consistency):
    """4: For each decile, show original MAE, debiased MAE, and gain."""
    dec_edges = np.percentile(epi, np.linspace(0, 100, 11))
    dec_centers = [(i * 10 + (i + 1) * 10) / 2 for i in range(10)]

    da_db = coord_a.copy()
    db_db = coord_b.copy()
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 3:
            continue
        da_db[m] -= (coord_a[m] - gt[m]).mean(axis=0)
        db_db[m] -= (coord_b[m] - gt[m]).mean(axis=0)

    rows = []
    for i in range(10):
        lo, hi = dec_edges[i], dec_edges[i + 1]
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 10:
            continue
        ca, cb, gm = coord_a[m], coord_b[m], gt[m]

        orig_der = float(np.linalg.norm(ca - gm, axis=1).mean())
        orig_gs  = float(np.linalg.norm(cb - gm, axis=1).mean())

        # per-axis baseline
        pa = _per_axis_fuse(ca, cb)
        pa_mae = float(np.linalg.norm(pa - gm, axis=1).mean())

        # full debias (upper bound)
        fd = (da_db[m] + db_db[m]) / 2.0
        fd_mae = float(np.linalg.norm(fd - gm, axis=1).mean())

        # consistency-gated
        cg = _per_axis_fuse(ca, cb)
        for j, (lo_j, hi_j) in enumerate(zip(edges[:-1], edges[1:])):
            cs = consistency[j]
            if cs is None:
                continue
            mm = (epi[m] >= lo_j) & (epi[m] < hi_j)
            if not mm.any():
                continue
            if cs["cos_sim_mean"] > 0.9:
                # debiased average on this ventile subset within decile
                cg[mm] = (da_db[m][mm] + db_db[m][mm]) / 2.0
        cg_mae = float(np.linalg.norm(cg - gm, axis=1).mean())

        rows.append({
            "pct_center": dec_centers[i],
            "n": int(m.sum()),
            "orig_DER": orig_der, "orig_gs": orig_gs,
            "F_per_axis": pa_mae,
            "G_full_debias": fd_mae,
            "E_consistency_gated": cg_mae,
        })

    return rows


def compute(stats: StatsAccumulator) -> dict:
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"error": "no epi_var_A"}

    epi = stats.epi_var_A.flatten()
    ca, cb, gt = stats.coord_A, stats.coord_B, stats.coord_gt

    # 1 & 2: calibration tables
    edges, strength, consistency = _build_ventile_tables(epi, ca, cb, gt)

    result = {
        "DER_mae": float(stats.error_A.mean()),
        "gs_mae":  float(stats.error_B.mean()),
        "avg_mae": float(np.linalg.norm((ca + cb) / 2.0 - gt, axis=1).mean()),
        "n":       int(len(epi)),
        "bias_strength_table":    strength,
        "direction_consistency_table": consistency,
    }

    # 3: consistency-gated debiasing
    result["E_consistency_gated"] = _consistency_gated(ca, cb, gt, epi, edges, consistency, thr=0.9)

    # F: per-axis baseline
    pa = _per_axis_fuse(ca, cb)
    result["F_per_axis_baseline"] = dict(zip(["mae", "rmse"], _mae_rmse(pa, gt)))

    # G: full-debias upper bound
    da, db = ca.copy(), cb.copy()
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 3: continue
        da[m] -= (ca[m] - gt[m]).mean(axis=0)
        db[m] -= (cb[m] - gt[m]).mean(axis=0)
    fd = (da + db) / 2.0
    result["G_full_debias"] = dict(zip(["mae", "rmse"], _mae_rmse(fd, gt)))

    # 4: per-decile verification
    result["per_decile"] = _per_decile_verification(ca, cb, gt, epi, edges, consistency)

    return result
