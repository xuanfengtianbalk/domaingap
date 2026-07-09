"""Fusion methods — comprehensive comparison.

Methods:
  Gated     — hard gate (p33/p66): low→Head1, mid→avg, high→Head2
  Avg       — simple equal-weight average
  Method A  — per_axis_trust: epi<tau→avg, else→(gs_x, gs_y, DER_z)
  Method B  — theta_gate:     theta<tau→avg, else→per_axis_select
  Method C  — table_debias:   ventile lookup table, subtract mean bias
  Method D  — epi_weighted:   w=epi/(epi+tau), fused=w·gs+(1-w)·DER
  BiasCorr  — alpha-sweep (kept for reference, not the focus)
"""

from __future__ import annotations
import numpy as np
from metrics.accumulator import StatsAccumulator


# ── helpers ──────────────────────────────────────────────────────────────────

def _mae_rmse(fused, gt):
    e = np.linalg.norm(fused - gt, axis=1)
    return float(e.mean()), float(np.sqrt((e ** 2).mean()))


def _per_axis_select(coord_a, coord_b, mask):
    """mask=True → per-axis select, mask=False → average."""
    fused = (coord_a + coord_b) / 2.0
    fused[mask, 0] = coord_b[mask, 0]
    fused[mask, 1] = coord_b[mask, 1]
    fused[mask, 2] = coord_a[mask, 2]
    return fused


def _sweep_best(sweep: dict, key="mae") -> dict:
    best_entry = min(sweep.values(), key=lambda v: v[key])
    return best_entry


# ── method A: per_axis_trust ─────────────────────────────────────────────────

def _per_axis_trust(coord_a, coord_b, gt, epi, taus):
    sweep = {}
    for tau in taus:
        mask_select = epi >= tau
        fused = _per_axis_select(coord_a, coord_b, mask_select)
        mae, rmse = _mae_rmse(fused, gt)
        sweep[str(tau)] = {
            "tau": float(tau),
            "mae":  mae,
            "rmse": rmse,
            "pct_above": float(mask_select.mean() * 100),
        }
    return sweep, _sweep_best(sweep)


# ── method B: theta_gate ─────────────────────────────────────────────────────

def _theta_gate(coord_a, coord_b, gt, epi, theta_thresholds):
    na = np.linalg.norm(coord_a - gt, axis=1)
    nb = np.linalg.norm(coord_b - gt, axis=1)
    denom = na * nb
    dot = ((coord_a - gt) * (coord_b - gt)).sum(axis=1)
    valid = denom > 1e-12
    cos_sim = np.ones_like(denom)
    cos_sim[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    theta = np.rad2deg(np.arccos(np.clip(cos_sim, -1.0, 1.0)))

    sweep = {}
    for thr in theta_thresholds:
        mask_select = theta >= thr
        fused = _per_axis_select(coord_a, coord_b, mask_select)
        mae, rmse = _mae_rmse(fused, gt)
        sweep[str(thr)] = {
            "theta_thr": float(thr),
            "mae":  mae,
            "rmse": rmse,
            "pct_above": float(mask_select.mean() * 100),
        }
    return sweep, _sweep_best(sweep)


# ── method C: table_debias ───────────────────────────────────────────────────

def _table_debias(coord_a, coord_b, gt, epi):
    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(np.round(edges, 8))

    t_der = np.zeros((len(edges) - 1, 3))
    t_gs  = np.zeros((len(edges) - 1, 3))
    bin_counts = np.zeros(len(edges) - 1)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (epi >= lo) & (epi < hi)
        n = m.sum()
        if n < 3:
            continue
        t_der[i] = (coord_a[m] - gt[m]).mean(axis=0)
        t_gs[i]  = (coord_b[m] - gt[m]).mean(axis=0)
        bin_counts[i] = n

    # Apply: for each pixel, find bin and subtract table
    da_corr = coord_a.copy()
    db_corr = coord_b.copy()
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_counts[i] < 3:
            continue
        m = (epi >= lo) & (epi < hi) if hi < np.inf else (epi >= lo)
        da_corr[m] -= t_der[i][None, :]
        db_corr[m] -= t_gs[i][None, :]

    mae_d, rmse_d = _mae_rmse(da_corr, gt)
    mae_g, rmse_g = _mae_rmse(db_corr, gt)
    orig_mae_a = float(np.linalg.norm(coord_a - gt, axis=1).mean())
    orig_mae_b = float(np.linalg.norm(coord_b - gt, axis=1).mean())

    return {
        "DER": {"mae": mae_d, "rmse": rmse_d, "gain": orig_mae_a - mae_d,
                "n_bins": int(sum(bin_counts > 3))},
        "gs":  {"mae": mae_g, "rmse": rmse_g, "gain": orig_mae_b - mae_g,
                "n_bins": int(sum(bin_counts > 3))},
    }


# ── method D: epi_weighted ───────────────────────────────────────────────────

def _epi_weighted(coord_a, coord_b, gt, epi, taus):
    sweep = {}
    for tau in taus:
        w = (epi[:, None] / (epi[:, None] + tau))
        fused = w * coord_b + (1.0 - w) * coord_a
        mae, rmse = _mae_rmse(fused, gt)
        sweep[str(tau)] = {
            "tau": float(tau),
            "mae":  mae,
            "rmse": rmse,
        }
    return sweep, _sweep_best(sweep)


# ── per-epi-decile MAE table (for summary plot) ──────────────────────────────

def _per_decile_mae(coord_a, coord_b, gt, epi, best_params):
    """Compute MAE for every method in each epi_std decile.

    Returns list of {pct_range, ...method_mae...}
    """
    edges = np.percentile(epi, np.linspace(0, 100, 11))
    names = [f"{lo}-{hi}" for lo, hi in zip(range(0, 100, 10), range(10, 110, 10))]

    rows = []
    for i, name in enumerate(names):
        lo, hi = edges[i], edges[i + 1]
        m = (epi >= lo) & (epi < hi) if i < 10 else (epi >= lo)
        n = m.sum()
        if n < 10:
            continue

        ca, cb, gt_m, epi_m = coord_a[m], coord_b[m], gt[m], epi[m]

        row = {"range": name, "center_pct": i * 10 + 5, "n": int(n),
               "head1_mae": float(np.linalg.norm(ca - gt_m, axis=1).mean()),
               "head2_mae": float(np.linalg.norm(cb - gt_m, axis=1).mean())}

        avg_f = (ca + cb) / 2.0
        row["avg_mae"] = float(np.linalg.norm(avg_f - gt_m, axis=1).mean())

        # Method A
        tau_a = best_params.get("per_axis_tau", 0.0005)
        fused_a = _per_axis_select(ca, cb, epi_m >= tau_a)
        row["per_axis_trust_mae"] = float(np.linalg.norm(fused_a - gt_m, axis=1).mean())

        # Method B
        na = np.linalg.norm(ca - gt_m, axis=1)
        nb = np.linalg.norm(cb - gt_m, axis=1)
        denom = na * nb
        dot = ((ca - gt_m) * (cb - gt_m)).sum(axis=1)
        valid = denom > 1e-12
        cos_sim = np.ones_like(denom)
        cos_sim[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
        theta = np.rad2deg(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        thr_b = best_params.get("theta_thr", 20.0)
        fused_b = _per_axis_select(ca, cb, theta >= thr_b)
        row["theta_gate_mae"] = float(np.linalg.norm(fused_b - gt_m, axis=1).mean())

        # Method D
        tau_d = best_params.get("epi_weighted_tau", 0.0005)
        w = epi_m[:, None] / (epi_m[:, None] + tau_d)
        fused_d = w * cb + (1.0 - w) * ca
        row["epi_weighted_mae"] = float(np.linalg.norm(fused_d - gt_m, axis=1).mean())

        rows.append(row)

    return rows


# ── main compute ─────────────────────────────────────────────────────────────

def compute(stats: StatsAccumulator) -> dict:
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

    # ---- shared tau values (concrete std, not percentiles) ----
    taus = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    theta_thresholds = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]

    # ---- method A: per_axis_trust ----
    sweep_a, best_a = _per_axis_trust(coord_a, coord_b, gt, epi, taus)
    result["per_axis_trust"] = {"sweep": sweep_a, "best": best_a}

    # ---- method B: theta_gate ----
    sweep_b, best_b = _theta_gate(coord_a, coord_b, gt, epi, theta_thresholds)
    result["theta_gate"] = {"sweep": sweep_b, "best": best_b}

    # ---- method C: table_debias ----
    result["table_debias"] = _table_debias(coord_a, coord_b, gt, epi)

    # ---- method D: epi_weighted ----
    sweep_d, best_d = _epi_weighted(coord_a, coord_b, gt, epi, taus)
    result["epi_weighted"] = {"sweep": sweep_d, "best": best_d}

    # ---- per-epi-decile MAE for summary visualization ----
    best_params = {
        "per_axis_tau":   float(best_a.get("tau", 0.0005)),
        "theta_thr":      float(best_b.get("theta_thr", 20.0)),
        "epi_weighted_tau": float(best_d.get("tau", 0.0005)),
    }
    result["per_decile"] = _per_decile_mae(coord_a, coord_b, gt, epi, best_params)

    return result
