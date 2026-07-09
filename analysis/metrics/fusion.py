"""Adaptive Residual Fusion (ARF) — the only recommended method.

  coord_fusion = coord_DER + alpha(epi_std) * (coord_GS - coord_DER)

alpha is read from a 20-ventile lookup table built from mean_gain
(E[error_DER - error_GS] | u_epi) — higher gain → trust GS more.
Zero training, zero extra parameters.
"""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator


def compute(stats: StatsAccumulator) -> dict:
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"error": "no epi_var_A"}

    epi = stats.epi_var_A.flatten()
    ca, cb, gt = stats.coord_A, stats.coord_B, stats.coord_gt
    ea, eb = stats.error_A, stats.error_B
    gain = ea - eb                                 # >0 → GS better

    # ── baselines ──
    result = {
        "DER_mae": float(ea.mean()),
        "gs_mae":  float(eb.mean()),
        "avg_mae": float(np.linalg.norm((ca + cb) / 2.0 - gt, axis=1).mean()),
        "n":       int(len(epi)),
    }

    # ── build 20-ventile alpha table from mean_gain ──
    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(np.round(edges, 8))
    n_bins = len(edges) - 1

    mean_gain = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:])):
        m = (epi >= lo) & (epi < hi)
        n = m.sum()
        if n < 3:
            continue
        mean_gain[i] = gain[m].mean()
        bin_counts[i] = n

    # alpha = normalized mean_gain  (clipped to [0, 1])
    gmax = mean_gain.max()
    alpha_table = np.clip(mean_gain / gmax, 0.0, 1.0) if gmax > 1e-12 else np.zeros(n_bins)

    # ── apply ARF ──
    delta = cb - ca                               # (N, 3)  residual
    fused = ca.copy()
    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:])):
        if bin_counts[i] < 3:
            continue
        m = (epi >= lo) & (epi < hi)
        fused[m] = ca[m] + alpha_table[i] * delta[m]

    e_fused = np.linalg.norm(fused - gt, axis=1)
    result["ARF_mae"]  = float(e_fused.mean())
    result["ARF_rmse"] = float(np.sqrt((e_fused ** 2).mean()))

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
            "mean_gain": float(mean_gain[i]),
            "alpha":     float(alpha_table[i]),
        })
    result["alpha_table"] = alpha_rows

    # ── per-decile MAE comparison ──
    dec_edges = np.percentile(epi, np.linspace(0, 100, 11))
    dec_centers = [(i * 10 + (i + 1) * 10) / 2 for i in range(10)]
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
        })
    result["per_decile"] = decile

    return result
