"""Visualization helpers for analysis."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics.accumulator import StatsAccumulator

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def plot_error_scatter(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Scatter plot of model A error vs model B error."""
    if len(stats.error_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(stats.error_A, stats.error_B, s=1, alpha=0.3)
    mx = max(stats.error_A.max(), stats.error_B.max())
    ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
    ax.set_xlabel("Model A error")
    ax.set_ylabel("Model B error")
    ax.set_title(f"{split}: error scatter (n={len(stats.error_A):,})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_scatter.png"), dpi=150)
    plt.close()


def plot_histogram(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Histogram of per-pixel errors for both models."""
    if len(stats.error_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    q = np.percentile(stats.error_A, 99)
    ax.hist(stats.error_A.clip(0, q), bins=50, alpha=0.5, label="Model A")
    ax.hist(stats.error_B.clip(0, q), bins=50, alpha=0.5, label="Model B")
    ax.set_xlabel("L2 Error")
    ax.legend()
    ax.set_title(f"{split}: error histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_hist.png"), dpi=150)
    plt.close()


def plot_epi_winrate(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Epistemic threshold vs Head2 win-rate curve.

    For each epi percentile (10% … 90%), keep only pixels ABOVE the
    threshold and compute Head2's win rate.
    """
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    thresholds = np.percentile(epi, np.arange(10, 100, 10))

    rates = []
    n_kept = []
    for thr in thresholds:
        mask = epi >= thr
        if mask.sum() > 0:
            rates.append(float((stats.error_A[mask] < stats.error_B[mask]).mean()))
        else:
            rates.append(float("nan"))
        n_kept.append(int(mask.sum()))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(thresholds, rates, "b-o", markersize=4)
    ax1.set_xlabel("Epistemic std threshold (>=)")
    ax1.set_ylabel("Head2 Win Rate", color="b")
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, np.array(n_kept) / len(epi) * 100, "r--", alpha=0.5)
    ax2.set_ylabel("Retained pixels %", color="r")
    ax2.set_ylim(0, 105)

    ax1.set_title(f"{split}: epi threshold → Head2 win-rate")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "epi_winrate.png"), dpi=150)
    plt.close()


def plot_calibration_error(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """MAE vs epistemic std for both models, with error bars per bin."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    ea = stats.error_A
    eb = stats.error_B
    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(edges)

    centers, mae_a, mae_b = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 10:
            continue
        centers.append((lo + hi) / 2)
        mae_a.append(ea[m].mean())
        mae_b.append(eb[m].mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(centers, mae_a, "b-o", markersize=4, label="Head1 (DER)")
    ax.plot(centers, mae_b, "r-s", markersize=4, label="Head2 (gs)")
    ax.fill_between(centers, mae_a, mae_b, where=np.array(mae_a) > np.array(mae_b),
                    alpha=0.15, color="red", label="Head2 better")
    ax.fill_between(centers, mae_a, mae_b, where=np.array(mae_a) <= np.array(mae_b),
                    alpha=0.15, color="blue", label="Head1 better")
    ax.set_xlabel("Epistemic std (epi_var_A)")
    ax.set_ylabel("MAE")
    ax.set_title(f"{split}: error vs epistemic uncertainty")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calibration_error.png"), dpi=150)
    plt.close()


def plot_calibration_gain(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Mean gain (Head1−Head2 error) and Head2 win-rate vs epistemic std."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    ea = stats.error_A
    eb = stats.error_B
    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(edges)

    centers, gains, winrates = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 10:
            continue
        centers.append((lo + hi) / 2)
        gains.append((ea[m] - eb[m]).mean())
        winrates.append(float((ea[m] < eb[m]).mean()))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(centers, gains, width=[c * 0.5 for c in np.diff(edges[:len(centers)+1])],
            alpha=0.5, color="steelblue", label="Mean gain (Head1−Head2)")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Epistemic std")
    ax1.set_ylabel("Mean Gain (>0 → Head2 better)", color="steelblue")
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.plot(centers, winrates, "r-o", markersize=5, label="Head2 win rate")
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Head2 Win Rate", color="red")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    ax1.set_title(f"{split}: gain & win rate vs epistemic uncertainty")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calibration_gain.png"), dpi=150)
    plt.close()


def plot_threshold_sweep(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Threshold τ scan: MAE of post-τ pixels and pixels-retained curve."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    ea = stats.error_A
    eb = stats.error_B
    n_total = len(epi)

    pcts = np.arange(50, 100, 2)
    τ_vals = np.percentile(epi, pcts)

    mae_a_arr, mae_b_arr, pct_retained = [], [], []
    for τ in τ_vals:
        m = epi >= τ
        mae_a_arr.append(ea[m].mean() if m.sum() > 0 else float("nan"))
        mae_b_arr.append(eb[m].mean() if m.sum() > 0 else float("nan"))
        pct_retained.append(m.sum() / n_total * 100)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(τ_vals, mae_a_arr, "b-o", markersize=4, label="Head1 MAE above τ")
    ax1.plot(τ_vals, mae_b_arr, "r-s", markersize=4, label="Head2 MAE above τ")
    ax1.set_xlabel("Epistemic threshold τ (keep pixels with epi_std ≥ τ)")
    ax1.set_ylabel("MAE of retained pixels")
    ax1.legend(fontsize=8, loc="center left")

    ax2 = ax1.twinx()
    ax2.fill_between(τ_vals, 0, pct_retained, alpha=0.15, color="gray")
    ax2.plot(τ_vals, pct_retained, "k--", alpha=0.5)
    ax2.set_ylabel("Pixels retained (%)", color="gray")

    ax1.set_title(f"{split}: threshold sweep — MAE of pixels above τ")
    ax1.set_xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "threshold_sweep.png"), dpi=150)
    plt.close()


# ── Fusion summary: per-epi-decile MAE — DER, gs, avg, ARF ─────────────────

def plot_fusion_summary(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    ca, cb, gt = stats.coord_A, stats.coord_B, stats.coord_gt
    ea, eb = stats.error_A, stats.error_B
    gain = ea - eb

    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(np.round(edges, 8))
    n_bins = len(edges) - 1
    mean_gain = np.zeros(n_bins)
    cnt = np.zeros(n_bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:])):
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 3: continue
        mean_gain[i] = gain[m].mean()
        cnt[i] = m.sum()
    gmax = mean_gain.max()
    alpha_table = np.clip(mean_gain / gmax, 0.0, 1.0) if gmax > 1e-12 else np.zeros(n_bins)

    delta = cb - ca
    arf_fused = ca.copy()
    for i, (lo, hi) in enumerate(zip(edges[:n_bins], edges[1:])):
        if cnt[i] < 3: continue
        m = (epi >= lo) & (epi < hi)
        arf_fused[m] = ca[m] + alpha_table[i] * delta[m]

    # ── H: Alpha-Joint Correction ──
    if stats.alea_var_A is not None and len(stats.alea_var_A) > 0:
        total_var = np.maximum(stats.epi_var_A.flatten() + stats.alea_var_A.flatten(), 0.0)
        total_std = np.sqrt(total_var)
    else:
        total_std = np.sqrt(np.maximum(stats.epi_var_A.flatten(), 0.0))

    ca_corr = ca.copy()
    eps = 1e-3
    for axis_idx in range(3):
        cj_vals = ca[:, axis_idx]
        gt_vals = gt[:, axis_idx]
        ts = total_std
        v = np.abs(cj_vals) >= eps
        if v.sum() < 10: continue
        cv, gv, tv = cj_vals[v], gt_vals[v], ts[v]
        alpha_v = (gv - cv) / (cv * tv + 1e-12)
        ts_edges = np.percentile(tv, np.linspace(0, 100, 11))
        p_min, p_max = cv.min(), cv.max()
        p_edges = np.arange(p_min, p_max + 0.025, 0.05)
        p_edges = np.unique(np.round(p_edges, 8))
        for tlo, thi in zip(ts_edges[:-1], ts_edges[1:]):
            m_t = (tv >= tlo) & (tv < thi)
            for plo, phi in zip(p_edges[:-1], p_edges[1:]):
                m_p = (cv >= plo) & (cv < phi)
                mc = m_t & m_p
                if mc.sum() < 5: continue
                ma = alpha_v[mc].mean()
                m_full = (ts >= tlo) & (ts < thi) & (cj_vals >= plo) & (cj_vals < phi)
                ca_corr[m_full, axis_idx] = cj_vals[m_full] + ma * cj_vals[m_full] * ts[m_full]

    dec_edges = np.percentile(epi, np.linspace(0, 100, 11))
    pct_c = [(i * 10 + (i + 1) * 10) / 2 for i in range(10)]

    mae = {"DER": [], "gs": [], "avg": [], "ARF": [], "H": []}
    for d in range(10):
        lo, hi = dec_edges[d], dec_edges[d + 1]
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 10:
            for k in mae: mae[k].append(float("nan"))
            continue
        ci, cj, gm = ca[m], cb[m], gt[m]
        mae["DER"].append(float(np.linalg.norm(ci - gm, axis=1).mean()))
        mae["gs"].append(float(np.linalg.norm(cj - gm, axis=1).mean()))
        mae["avg"].append(float(np.linalg.norm((ci + cj) / 2.0 - gm, axis=1).mean()))
        mae["ARF"].append(float(np.linalg.norm(arf_fused[m] - gm, axis=1).mean()))
        mae["H"].append(float(np.linalg.norm(ca_corr[m] - gm, axis=1).mean()))

    colors = {"DER": "#1f77b4", "gs": "#d62728", "avg": "#7f7f7f", "ARF": "#2ca02c", "H": "#9467bd"}
    labels = {"DER": "DER", "gs": "gs", "avg": "avg", "ARF": "ARF (gain-lookup)", "H": "H: alpha-joint correct"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for k in ["DER", "gs", "avg", "ARF", "H"]:
        vals = mae[k]
        if all(np.isnan(v) for v in vals): continue
        ax.plot(pct_c, vals, "o-", color=colors[k], markersize=5, linewidth=1.5, label=labels[k])
    ax.set_xlabel("Epistemic Percentile")
    ax.set_ylabel("MAE")
    ax.set_xlim(0, 100)
    ax.set_title(f"{split}: Adaptive Residual Fusion")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fusion_summary.png"), dpi=200)
    plt.close()


# ── Per-axis bias analysis (x/y/z, 100 percentile bins) ────────────────────

_axes_names = ["x", "y", "z"]
_axis_colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}


def plot_sign_agreement(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Fraction of same-sign bias per epi percentile, 3 lines (x/y/z)."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    da, db = stats.diff_A, stats.diff_B

    edges = np.percentile(epi, np.linspace(0, 100, 101))
    edges = np.unique(np.round(edges, 10))
    n_bins = len(edges) - 1
    centers = [(i + 0.5) * (100.0 / n_bins) for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(8, 4))
    for a, idx in [("x", 0), ("y", 1), ("z", 2)]:
        fracs = []
        for lo, hi in zip(edges[:n_bins], edges[1:]):
            m = (epi >= lo) & (epi < hi)
            if m.sum() < 3:
                fracs.append(float("nan"))
                continue
            same = ((da[m, idx] > 0) & (db[m, idx] > 0)) | ((da[m, idx] < 0) & (db[m, idx] < 0))
            fracs.append(float(same.mean()))
        ax.plot(centers, fracs, "-", color=_axis_colors[a], linewidth=1.5, label=a)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epistemic Percentile")
    ax.set_ylabel("Fraction same sign")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{split}: bias sign agreement vs epi")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sign_agreement.png"), dpi=200)
    plt.close()


def plot_bias_magnitude(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """|bias| median + p25-p75 band vs epi percentile, 2 panels (DER/gs), 3 lines each."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    da, db = stats.diff_A, stats.diff_B

    edges = np.percentile(epi, np.linspace(0, 100, 101))
    edges = np.unique(np.round(edges, 10))
    n_bins = len(edges) - 1
    centers = [(i + 0.5) * (100.0 / n_bins) for i in range(n_bins)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True, sharey=True)
    for ax, label, diff in [(ax1, "DER", da), (ax2, "gs", db)]:
        for a, idx in [("x", 0), ("y", 1), ("z", 2)]:
            med, lo, hi = [], [], []
            for lo_e, hi_e in zip(edges[:n_bins], edges[1:]):
                m = (epi >= lo_e) & (epi < hi_e)
                n = m.sum()
                if n < 3:
                    med.append(float("nan")); lo.append(float("nan")); hi.append(float("nan"))
                    continue
                vals = np.abs(diff[m, idx])
                med.append(float(np.median(vals)))
                lo.append(float(np.percentile(vals, 25)))
                hi.append(float(np.percentile(vals, 75)))
            ax.plot(centers, med, "-", color=_axis_colors[a], linewidth=1.2, label=a)
            ax.fill_between(centers, lo, hi, color=_axis_colors[a], alpha=0.12)
        ax.set_title(f"|bias_{{{label}}}|")
        ax.set_xlabel("Epistemic Percentile")
        ax.set_ylabel("|bias| (median, p25-p75)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"{split}: bias magnitude vs epi", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_magnitude.png"), dpi=200)
    plt.close()


def plot_disagreement_ratio(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """|DER-gs|/|bias_DER| and |DER-gs|/|bias_gs| median vs epi, 2 panels."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    da, db = stats.diff_A, stats.diff_B
    eps = 1e-12

    edges = np.percentile(epi, np.linspace(0, 100, 101))
    edges = np.unique(np.round(edges, 10))
    n_bins = len(edges) - 1
    centers = [(i + 0.5) * (100.0 / n_bins) for i in range(n_bins)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)
    for ax, label, ref in [(ax1, "|DER-gs| / |bias_DER|", da), (ax2, "|DER-gs| / |bias_gs|", db)]:
        for a, idx in [("x", 0), ("y", 1), ("z", 2)]:
            med = []
            for lo, hi in zip(edges[:n_bins], edges[1:]):
                m = (epi >= lo) & (epi < hi)
                if m.sum() < 3:
                    med.append(float("nan"))
                    continue
                ratio = np.abs(da[m, idx] - db[m, idx]) / (np.abs(ref[m, idx]) + eps)
                med.append(float(np.median(ratio)))
            ax.plot(centers, med, "-", color=_axis_colors[a], linewidth=1.5, label=a)
        ax.set_title(label)
        ax.set_xlabel("Epistemic Percentile")
        ax.set_ylabel("Median ratio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"{split}: disagreement ratio vs epi", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "disagreement_ratio.png"), dpi=200)
    plt.close()


def plot_gt_conditioned(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Mean prediction + p25-p75 band vs GT coordinate, per axis. Identity line = perfect."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    ca, cb, gt = stats.coord_A, stats.coord_B, stats.coord_gt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, name, idx in zip(axes, _axes_names, [0, 1, 2]):
        gt_vals = gt[:, idx]
        gt_edges = np.percentile(gt_vals, np.linspace(0, 100, 21))
        gtm, ma, p25_a, p75_a, p50_a, mb, p25_b, p75_b, p50_b = [], [], [], [], [], [], [], [], []
        for lo, hi in zip(gt_edges[:-1], gt_edges[1:]):
            m = (gt_vals >= lo) & (gt_vals < hi)
            if m.sum() < 3:
                continue
            gtm.append(float(gt_vals[m].mean()))
            ma.append(float(ca[m, idx].mean()))
            p50_a.append(float(np.percentile(ca[m, idx], 50)))
            p25_a.append(float(np.percentile(ca[m, idx], 25)))
            p75_a.append(float(np.percentile(ca[m, idx], 75)))
            mb.append(float(cb[m, idx].mean()))
            p50_b.append(float(np.percentile(cb[m, idx], 50)))
            p25_b.append(float(np.percentile(cb[m, idx], 25)))
            p75_b.append(float(np.percentile(cb[m, idx], 75)))
        ax.fill_between(gtm, p25_a, p75_a, color="#1f77b4", alpha=0.08)
        ax.fill_between(gtm, p25_b, p75_b, color="#d62728", alpha=0.08)
        ax.plot(gtm, ma,   "o-", color="#1f77b4", markersize=3, linewidth=1.5, label="DER mean")
        ax.plot(gtm, p50_a, "o--", color="#1f77b4", markersize=2, linewidth=0.8, alpha=0.5, label="DER med")
        ax.plot(gtm, mb,   "s-", color="#d62728", markersize=3, linewidth=1.5, label="gs mean")
        ax.plot(gtm, p50_b, "s--", color="#d62728", markersize=2, linewidth=0.8, alpha=0.5, label="gs med")
        ax.set_ylabel(f"pred_{name} (— mean, -- med)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"{split}: prediction vs GT coordinate", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gt_conditioned.png"), dpi=200)
    plt.close()


def plot_ct_vs_uncertainty(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Per GT bin: total_std (x) vs bias (y) curves. Fixed-step GT bins from config."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    gr = cfg["gt_ranges"]
    step = gr["bin_step"]

    if stats.alea_var_A is not None and len(stats.alea_var_A) > 0:
        total_std = np.sqrt(np.maximum(stats.epi_var_A.flatten() + stats.alea_var_A.flatten(), 0.0))
    else:
        total_std = np.sqrt(np.maximum(stats.epi_var_A.flatten(), 0.0))
    ca, cb, gt = stats.coord_A, stats.coord_B, stats.coord_gt

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    keys = ["x", "y", "z"]
    for ax, key, idx in zip(axes, keys, [0, 1, 2]):
        gt_vals = gt[:, idx]
        g_range = gr[key]
        gt_edges = np.arange(g_range[0], g_range[1] + step * 0.5, step)
        n_gt_bins = len(gt_edges) - 1
        # Pick 5 representative GT bins
        pick_idx = [0, n_gt_bins // 4, n_gt_bins // 2, 3 * n_gt_bins // 4, n_gt_bins - 1]

        colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(pick_idx)))

        for pi, gi in enumerate(pick_idx):
            if gi >= n_gt_bins:
                continue
            lo, hi = gt_edges[gi], gt_edges[gi + 1]
            m_gt = (gt_vals >= lo) & (gt_vals < hi)
            if m_gt.sum() < 30:
                continue
            ts, cs, cbs = total_std[m_gt], ca[m_gt, idx], cb[m_gt, idx]
            gs_sub = gt_vals[m_gt]

            ts_edges = np.percentile(ts, np.linspace(0, 100, 11))
            ts_ctr, ba_mean, ba_med, bb_mean, bb_med = [], [], [], [], []
            for elo, ehi in zip(ts_edges[:-1], ts_edges[1:]):
                me = (ts >= elo) & (ts < ehi)
                if me.sum() < 5:
                    continue
                ts_ctr.append(float(ts[me].mean()))
                delta_a = cs[me] - gs_sub[me]
                delta_b = cbs[me] - gs_sub[me]
                ba_mean.append(float(delta_a.mean()))
                ba_med.append(float(np.percentile(delta_a, 50)))
                bb_mean.append(float(delta_b.mean()))
                bb_med.append(float(np.percentile(delta_b, 50)))

            gt_mid = float(gs_sub.mean())
            ax.plot(ts_ctr, ba_mean, "-", color=colors[pi], linewidth=1.2,
                    label=f"GT≈{gt_mid:+.2f}")
            ax.plot(ts_ctr, ba_med, "--", color=colors[pi], linewidth=0.7, alpha=0.5)
            ax.plot(ts_ctr, bb_mean, "-.", color=colors[pi], linewidth=0.8, alpha=0.6)
            ax.plot(ts_ctr, bb_med, ":", color=colors[pi], linewidth=0.6, alpha=0.4)

        ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
        ax.set_title(f"GT_{key}")
        ax.set_xlabel("mean total_std")
        ax.set_ylabel("bias (— DER mean, -- DER med, -. gs mean, ·· gs med)")
        ax.set_xscale("log")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.15)

    fig.suptitle(f"{split}: total_std vs bias per GT bin  (colored by GT position)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ct_vs_uncertainty.png"), dpi=200)
    plt.close()


def plot_alpha_correct_vs_unc(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """DER vs H (alpha-joint correct) MAE per total_std bin.
    Two panels: percentile x-axis + actual value x-axis."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    if stats.alea_var_A is not None and len(stats.alea_var_A) > 0:
        total_std = np.sqrt(np.maximum(stats.epi_var_A.flatten() + stats.alea_var_A.flatten(), 0.0))
    else:
        total_std = np.sqrt(np.maximum(stats.epi_var_A.flatten(), 0.0))
    ca, gt = stats.coord_A, stats.coord_gt

    # Compute H (same logic as fusion.py)
    ca_corr = ca.copy()
    eps = 1e-3
    for axis_idx in range(3):
        cj_vals = ca[:, axis_idx]; gv_vals = gt[:, axis_idx]; ts = total_std
        v = np.abs(cj_vals) >= eps
        if v.sum() < 10: continue
        cv, gv, tv = cj_vals[v], gv_vals[v], ts[v]
        alpha_v = (gv - cv) / (cv * tv + 1e-12)
        ts_edges = np.percentile(tv, np.linspace(0, 100, 11))
        p_min, p_max = cv.min(), cv.max()
        p_edges = np.arange(p_min, p_max + 0.025, 0.05)
        for tlo, thi in zip(ts_edges[:-1], ts_edges[1:]):
            m_t = (tv >= tlo) & (tv < thi)
            for plo, phi in zip(p_edges[:-1], p_edges[1:]):
                m_p = (cv >= plo) & (cv < phi)
                mc = m_t & m_p
                if mc.sum() < 5: continue
                ma = alpha_v[mc].mean()
                m_full = (ts >= tlo) & (ts < thi) & (cj_vals >= plo) & (cj_vals < phi)
                ca_corr[m_full, axis_idx] = cj_vals[m_full] + ma * cj_vals[m_full] * ts[m_full]

    # Bin by total_std (10 percentile bins, same as CSV)
    ts_edges = np.percentile(total_std, np.linspace(0, 100, 11))
    ts_centers_pct = [(i + 0.5) * 10 for i in range(10)]
    ts_centers_val = []
    der_mae, h_mae = [], []

    for lo, hi in zip(ts_edges[:-1], ts_edges[1:]):
        m = (total_std >= lo) & (total_std < hi)
        if m.sum() < 10:
            continue
        ts_centers_val.append(float(total_std[m].mean()))
        der_mae.append(float(np.linalg.norm(ca[m] - gt[m], axis=1).mean()))
        h_mae.append(float(np.linalg.norm(ca_corr[m] - gt[m], axis=1).mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.plot(ts_centers_pct, der_mae, "bo-", markersize=5, linewidth=1.5, label="DER")
    ax1.plot(ts_centers_pct, h_mae,  "mo-", markersize=5, linewidth=1.5, label="H (alpha-correct)")
    ax1.set_xlabel("total_std percentile")
    ax1.set_ylabel("MAE")
    ax1.set_title(f"{split}: MAE per total_std percentile")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, 100)

    ax2.plot(ts_centers_val, der_mae, "bo-", markersize=5, linewidth=1.5, label="DER")
    ax2.plot(ts_centers_val, h_mae,  "mo-", markersize=5, linewidth=1.5, label="H (alpha-correct)")
    ax2.set_xlabel("mean total_std")
    ax2.set_ylabel("MAE")
    ax2.set_title(f"{split}: MAE per total_std value")
    ax2.set_xscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alpha_correct_vs_unc.png"), dpi=200)
    plt.close()
