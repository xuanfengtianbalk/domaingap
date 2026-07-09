"""Gated fusion: low-epi -> Head1, mid -> average, high -> Head2.

Also sweeps bias correction: both models corrected by epi_std-weighted
disagreement direction, since both share the same-direction bias.
"""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator


def _bias_corrected_error(coord, gt, epi, alpha, dir_vec):
    corr = coord - alpha * epi[:, None] * dir_vec
    e = np.linalg.norm(corr - gt, axis=1)
    return float(e.mean()), float(np.sqrt((e ** 2).mean()))


def _epi_level_breakdown(coord_a, coord_b, gt, epi, ea, eb, dir_vec, alphas,
                         label_keys, pcts, edges, epi_levels):
    lev = {}
    for name, lo, hi in epi_levels:
        m = (epi >= lo) & (epi < hi) if hi < np.inf else (epi >= lo)
        if m.sum() < 10:
            continue
        entry = {
            "epi_lo": float(lo),
            "epi_hi": float(hi) if hi < np.inf else "inf",
            "n": int(m.sum()),
            "original_mae_DER": float(ea[m].mean()),
            "original_mae_gs":  float(eb[m].mean()),
            "bias_DER": {},
            "bias_gs": {},
        }
        for a, a_key in zip(alphas, label_keys):
            mae_d, _ = _bias_corrected_error(coord_a[m], gt[m], epi[m], a, dir_vec[m])
            mae_g, _ = _bias_corrected_error(coord_b[m], gt[m], epi[m], a, dir_vec[m])
            entry["bias_DER"][a_key] = {
                "mae": mae_d,
                "gain": entry["original_mae_DER"] - mae_d,
            }
            entry["bias_gs"][a_key] = {
                "mae": mae_g,
                "gain": entry["original_mae_gs"] - mae_g,
            }
        lev[name] = entry
    return lev


def compute(stats: StatsAccumulator) -> dict:
    """Simple uncertainty-gated fusion and constant-weight baselines.

    Uses Head1's epistemic uncertainty to select:
      - low_epi (< p33)           -> Head1 only
      - mid_epi ([p33, p66))      -> equal-weight average
      - high_epi (>= p66)         -> Head2 only

    Also reports constant-weight fusion baselines for comparison.
    """
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return {"gated_mae": None, "gated_rmse": None}

    epi = stats.epi_var_A.flatten()
    lo, hi = np.percentile(epi, [33, 66])

    coord_a = stats.coord_A  # (N, 3)
    coord_b = stats.coord_B
    gt = stats.coord_gt

    # ---- gated fusion ----
    fused = np.where(
        epi[:, None] < lo, coord_a,
        np.where(epi[:, None] < hi, (coord_a + coord_b) / 2.0, coord_b),
    )
    fe = np.linalg.norm(fused - gt, axis=1)

    # ---- baselines ----
    avg = (coord_a + coord_b) / 2.0
    avg_e = np.linalg.norm(avg - gt, axis=1)

    result = {
        "gated_mae":       float(fe.mean()),
        "gated_rmse":      float(np.sqrt((fe ** 2).mean())),
        "avg_mae":         float(avg_e.mean()),
        "avg_rmse":        float(np.sqrt((avg_e ** 2).mean())),
        "head1_mae":       float(stats.error_A.mean()),
        "head1_rmse":      float(np.sqrt((stats.error_A ** 2).mean())),
        "head2_mae":       float(stats.error_B.mean()),
        "head2_rmse":      float(np.sqrt((stats.error_B ** 2).mean())),
        "n":               int(len(stats.error_A)),
    }

    # ---- bias correction sweep ----
    dir_vec = coord_a - coord_b                                          # (N, 3)
    alphas = np.arange(0.005, 0.055, 0.005)
    label_keys = [f"{a:.3f}" for a in alphas]
    bias_der = {}
    bias_gs = {}
    best_der = {"alpha": None, "mae": float("inf")}
    best_gs  = {"alpha": None, "mae": float("inf")}

    for a, a_key in zip(alphas, label_keys):
        mae_d, rmse_d = _bias_corrected_error(coord_a, gt, epi, a, dir_vec)
        mae_g, rmse_g = _bias_corrected_error(coord_b, gt, epi, a, dir_vec)
        bias_der[a_key] = {"mae": mae_d, "rmse": rmse_d}
        bias_gs[a_key]  = {"mae": mae_g, "rmse": rmse_g}
        if mae_d < best_der["mae"]:
            best_der = {"alpha": float(a), "mae": mae_d, "rmse": rmse_d,
                        "gain": result["head1_mae"] - mae_d}
        if mae_g < best_gs["mae"]:
            best_gs  = {"alpha": float(a), "mae": mae_g, "rmse": rmse_g,
                        "gain": result["head2_mae"] - mae_g}

    result["bias_DER"] = {"sweep": bias_der, "best": best_der}
    result["bias_gs"]  = {"sweep": bias_gs,  "best": best_gs}

    # ---- per-epi-level breakdown ----
    pcts = [10, 30, 50, 70, 90, 99]
    edges = np.percentile(epi, pcts)
    all_edges = np.concatenate([[0.0], edges, [float("inf")]])
    names = ["p0-p10", "p10-p30", "p30-p50", "p50-p70", "p70-p90", "p90-p99", "p99-p100"]
    levels = []
    for i, name in enumerate(names):
        levels.append((name, all_edges[i], all_edges[i + 1]))

    result["by_epi_level"] = _epi_level_breakdown(
        coord_a, coord_b, gt, epi, stats.error_A, stats.error_B,
        dir_vec, alphas, label_keys, pcts, edges, levels,
    )

    return result
