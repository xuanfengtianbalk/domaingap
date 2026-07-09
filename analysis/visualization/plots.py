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

    dec_edges = np.percentile(epi, np.linspace(0, 100, 11))
    pct_c = [(i * 10 + (i + 1) * 10) / 2 for i in range(10)]

    mae = {"DER": [], "gs": [], "avg": [], "ARF": []}
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

    colors = {"DER": "#1f77b4", "gs": "#d62728", "avg": "#7f7f7f", "ARF": "#2ca02c"}
    labels = {"DER": "DER", "gs": "gs", "avg": "avg", "ARF": "ARF (gain-lookup)"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for k in ["DER", "gs", "avg", "ARF"]:
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


# ---------------------------------------------------------------------------
# Paper-ready bias analysis figures
# ---------------------------------------------------------------------------

def _decile_pearson(epi, diff_a, diff_b):
    """Compute Pearson r per decile. Returns (pct_centers, r_values)."""
    edges = np.percentile(epi, np.linspace(0, 100, 11))
    centers, r_vals = [], []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        m = (epi >= lo) & (epi < hi) if i < 10 else (epi >= lo)
        if m.sum() < 3:
            continue
        bias_a = np.linalg.norm(diff_a[m], axis=1)
        bias_b = np.linalg.norm(diff_b[m], axis=1)
        c = np.corrcoef(bias_a, bias_b)
        centers.append((i + 0.5) * 10)
        r_vals.append(float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0)
    return centers, r_vals


def _ventile_means(epi, diff_a, diff_b, signed=False):
    """Compute per-ventile mean of |bias| or signed bias_x/y/z. Returns (centers, ...data)."""
    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(edges)
    pct_edges_20 = np.linspace(0, 100, 21)
    centers = []
    ba, bb, bax, bay, baz, bbx, bby, bbz = [], [], [], [], [], [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 10:
            continue
        center_pct = (pct_edges_20[i] + pct_edges_20[i + 1]) / 2
        actual_pct_lo, actual_pct_hi = pct_edges_20[i], pct_edges_20[i + 1]
        centers.append((actual_pct_lo + actual_pct_hi) / 2)
        ba.append(float(np.linalg.norm(diff_a[m], axis=1).mean()))
        bb.append(float(np.linalg.norm(diff_b[m], axis=1).mean()))
        if signed:
            bax.append(float(diff_a[m, 0].mean()))
            bay.append(float(diff_a[m, 1].mean()))
            baz.append(float(diff_a[m, 2].mean()))
            bbx.append(float(diff_b[m, 0].mean()))
            bby.append(float(diff_b[m, 1].mean()))
            bbz.append(float(diff_b[m, 2].mean()))
    data = (ba, bb)
    if signed:
        data = (ba, bb, bax, bay, baz, bbx, bby, bbz)
    return centers, data


# ---- Figure 1: Pearson r vs Epistemic percentile (descending) ----

def plot_bias_pearson_vs_epi(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    if len(stats.diff_A) == 0 or stats.epi_var_A is None:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)
    epi = stats.epi_var_A.flatten()
    centers, r_vals = _decile_pearson(epi, stats.diff_A, stats.diff_B)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(centers, r_vals, "ko-", markersize=6, linewidth=1.5)
    ax.set_xlabel("Epistemic Percentile")
    ax.set_ylabel("Pearson r  (|bias_DER|, |bias_gs|)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 100)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4)
    ax.set_title(f"{split}: bias correlation decays with uncertainty")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_pearson_vs_epi.png"), dpi=200)
    plt.close()


# ---- Figure 2: |Bias| vs Epistemic percentile (ascending) ----

def plot_bias_magnitude_vs_epi(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    if len(stats.diff_A) == 0 or stats.epi_var_A is None:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)
    epi = stats.epi_var_A.flatten()
    centers, (ba, bb) = _ventile_means(epi, stats.diff_A, stats.diff_B)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(centers, ba, "bo-", markersize=4, linewidth=1.2, label="E(|bias_DER|)")
    ax.plot(centers, bb, "rs-", markersize=4, linewidth=1.2, label="E(|bias_gs|)")
    ax.set_xlabel("Epistemic Percentile")
    ax.set_ylabel("Mean  |bias|")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8)
    ax.set_title(f"{split}: bias magnitude grows with uncertainty")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_magnitude_vs_epi.png"), dpi=200)
    plt.close()


# ---- Figure 3: Bias Drift (signed bias x/y/z, DER only) ----

def plot_bias_drift(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    if len(stats.diff_A) == 0 or stats.epi_var_A is None:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)
    epi = stats.epi_var_A.flatten()
    centers, data = _ventile_means(epi, stats.diff_A, stats.diff_B, signed=True)
    _, _, bax, bay, baz, bbx, bby, bbz = data

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
    for axis, va, vb in [("x", bax, bbx), ("y", bay, bby), ("z", baz, bbz)]:
        c = colors[axis]
        ax.plot(centers, va, "-o", color=c, markersize=4, linewidth=1.2,
                label=f"DER bias_{axis}")
        ax.plot(centers, vb, "--s", color=c, markersize=3, linewidth=0.8, alpha=0.5)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Epistemic Percentile")
    ax.set_ylabel("E(signed bias)")
    ax.set_xlim(0, 100)
    ax.set_title(f"{split}: bias direction drifts with uncertainty  (— DER, -- gs)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_drift.png"), dpi=200)
    plt.close()


# ---- Figure 4: Combined dual-axis ----

def plot_bias_combined(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    if len(stats.diff_A) == 0 or stats.epi_var_A is None:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)
    epi = stats.epi_var_A.flatten()

    # Pearson on decile centers
    dec_centers, r_vals = _decile_pearson(epi, stats.diff_A, stats.diff_B)

    # Mean |bias| on decile centers (recompute at decile for alignment)
    dec_edges = np.percentile(epi, np.linspace(0, 100, 11))
    dec_bias_a, dec_bias_b = [], []
    for i in range(10):
        lo, hi = dec_edges[i], dec_edges[i + 1]
        m = (epi >= lo) & (epi < hi) if i < 10 else (epi >= lo)
        if m.sum() < 10:
            continue
        dec_bias_a.append(float(np.linalg.norm(stats.diff_A[m], axis=1).mean()))
        dec_bias_b.append(float(np.linalg.norm(stats.diff_B[m], axis=1).mean()))

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Left y-axis: Pearson r
    ax1.plot(dec_centers, r_vals, "ko-", markersize=7, linewidth=2, label="Pearson r")
    ax1.set_xlabel("Epistemic Percentile", fontsize=12)
    ax1.set_ylabel("Pearson  r", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 100)
    ax1.tick_params(axis="y")

    # Right y-axis: |bias|
    ax2 = ax1.twinx()
    mean_bias = [(a + b) / 2 for a, b in zip(dec_bias_a, dec_bias_b)]
    ax2.plot(dec_centers, dec_bias_a, "bs-", markersize=5, linewidth=1.2, alpha=0.7, label="|bias_DER|")
    ax2.plot(dec_centers, dec_bias_b, "rD-", markersize=5, linewidth=1.2, alpha=0.7, label="|bias_gs|")
    ax2.set_ylabel("Mean  |bias|", fontsize=12)
    ax2.tick_params(axis="y")

    # Shared legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    ax1.set_title(f"{split}: correlation decays, bias grows with uncertainty", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_combined.png"), dpi=200)
    plt.close()


# ---- Figure 5: Bias Angle theta = arccos(e_DER · e_GS / |e_DER||e_GS|) ----

def plot_bias_angle_vs_epi(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    if len(stats.diff_A) == 0 or stats.epi_var_A is None:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    da, db = stats.diff_A, stats.diff_B
    na = np.linalg.norm(da, axis=1)
    nb = np.linalg.norm(db, axis=1)
    valid = (na > 1e-12) & (nb > 1e-12)
    cos_sim = np.full_like(na, np.nan)
    cos_sim[valid] = np.clip((da[valid] * db[valid]).sum(axis=1) / (na[valid] * nb[valid]), -1.0, 1.0)
    angle = np.rad2deg(np.arccos(np.clip(cos_sim, -1.0, 1.0)))

    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(edges)

    centers, angles = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (epi >= lo) & (epi < hi)
        n = m.sum()
        if n < 10:
            continue
        vm = m & valid
        if vm.sum() < 3:
            continue
        centers.append((lo + hi) / 2)
        angles.append(float(np.mean(angle[vm])))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(centers, angles, "o-", color="darkred", markersize=4, linewidth=1.5)
    ax.set_xlabel("Epistemic Percentile")
    ax.set_ylabel("Mean θ  (degrees)")
    ax.set_xlim(0, 100)
    ax.set_title(f"{split}: error-vector angle grows with uncertainty")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_angle_vs_epi.png"), dpi=200)
    plt.close()
