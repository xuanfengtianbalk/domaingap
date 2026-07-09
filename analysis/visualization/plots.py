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
    """Mean prediction vs GT coordinate, per axis. Identity line = perfect."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    ca, cb, gt = stats.coord_A, stats.coord_B, stats.coord_gt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, name, idx in zip(axes, _axes_names, [0, 1, 2]):
        gt_vals = gt[:, idx]
        gt_edges = np.percentile(gt_vals, np.linspace(0, 100, 21))
        gtm, pred_a, pred_b = [], [], []
        for lo, hi in zip(gt_edges[:-1], gt_edges[1:]):
            m = (gt_vals >= lo) & (gt_vals < hi)
            if m.sum() < 3:
                continue
            gtm.append(float(gt_vals[m].mean()))
            pred_a.append(float(ca[m, idx].mean()))
            pred_b.append(float(cb[m, idx].mean()))
        ax.plot(gtm, pred_a, "bo-", markersize=4, label="DER")
        ax.plot(gtm, pred_b, "rs-", markersize=4, label="gs")
        mx = max(abs(np.array(gtm).min()), abs(np.array(gtm).max())) * 1.1
        ax.plot([-mx, mx], [-mx, mx], "gray", linestyle=":", alpha=0.5)
        ax.set_title(f"GT_{name}")
        ax.set_xlabel(f"GT_{name}")
        ax.set_ylabel(f"E(pred_{name})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"{split}: prediction vs GT coordinate", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gt_conditioned.png"), dpi=200)
    plt.close()
