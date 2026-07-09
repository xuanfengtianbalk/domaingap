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


def plot_bias_correction(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Bias correction sweep: DER / gs MAE vs alpha with epi_std-scaled correction."""
    if stats.epi_var_A is None or len(stats.epi_var_A) == 0:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    coord_a = stats.coord_A
    coord_b = stats.coord_B
    gt = stats.coord_gt
    dir_vec = coord_a - coord_b
    orig_mae_a = stats.error_A.mean()
    orig_mae_b = stats.error_B.mean()

    alphas = np.arange(0.0, 0.055, 0.005)
    mae_a_list, mae_b_list = [orig_mae_a], [orig_mae_b]

    for a in alphas[1:]:
        corr_a = coord_a - a * epi[:, None] * dir_vec
        corr_b = coord_b - a * epi[:, None] * dir_vec
        mae_a_list.append(float(np.linalg.norm(corr_a - gt, axis=1).mean()))
        mae_b_list.append(float(np.linalg.norm(corr_b - gt, axis=1).mean()))

    best_idx_a = int(np.argmin(mae_a_list))
    best_idx_b = int(np.argmin(mae_b_list))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(alphas, mae_a_list, "b-o", markersize=5, label="DER bias-corrected")
    ax1.plot(alphas, mae_b_list, "r-s", markersize=5, label="gs bias-corrected")
    ax1.axhline(orig_mae_a, color="b", linestyle="--", alpha=0.3, label=f"DER original ({orig_mae_a:.4f})")
    ax1.axhline(orig_mae_b, color="r", linestyle="--", alpha=0.3, label=f"gs original ({orig_mae_b:.4f})")
    ax1.axvline(alphas[best_idx_a], color="b", linestyle=":", alpha=0.6,
                label=f"best DER alpha={alphas[best_idx_a]:.3f}")
    ax1.axvline(alphas[best_idx_b], color="r", linestyle=":", alpha=0.6,
                label=f"best gs  alpha={alphas[best_idx_b]:.3f}")
    ax1.set_xlabel("Bias correction alpha")
    ax1.set_ylabel("MAE")
    ax1.set_title(f"{split}: bias correction sweep (DER/gs - alpha * std * dir)")
    ax1.legend(fontsize=7, loc="center right")
    ax1.set_xlim(0, 0.05)

    ax2 = ax1.twinx()
    gain_a = [orig_mae_a - m for m in mae_a_list]
    gain_b = [orig_mae_b - m for m in mae_b_list]
    ax2.plot(alphas, gain_a, "b--", alpha=0.4, markersize=0)
    ax2.plot(alphas, gain_b, "r--", alpha=0.4, markersize=0)
    ax2.set_ylabel("Gain (MAE improvement)", color="gray")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_correction.png"), dpi=150)
    plt.close()


def plot_bias_corr_scatter(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Scatter: bias_DER vs bias_gs, colored by epi_std decile."""
    if len(stats.diff_A) == 0 or stats.epi_var_A is None:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    bias_a = np.linalg.norm(stats.diff_A, axis=1)
    bias_b = np.linalg.norm(stats.diff_B, axis=1)
    epi = stats.epi_var_A.flatten()
    edges = np.percentile(epi, np.linspace(0, 100, 11))

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 10))
    for i in range(10):
        m = (epi >= edges[i]) & (epi < edges[i + 1]) if i < 9 else (epi >= edges[i])
        if m.sum() < 50:
            continue
        ax.scatter(bias_a[m], bias_b[m], s=0.5, alpha=0.3, color=colors[i],
                   label=f"p{i*10}-p{(i+1)*10}" if i in [0, 4, 9] else "",
                   rasterized=True)

    mx = max(bias_a.max(), bias_b.max())
    ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
    ax.set_xlabel("|bias| DER")
    ax.set_ylabel("|bias| gs")
    ax.set_title(f"{split}: bias DER vs bias gs (colored by epi decile)")
    ax.legend(fontsize=7, loc="lower right", markerscale=5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_corr_scatter.png"), dpi=150)
    plt.close()


def plot_bias_mag_vs_epi(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Line plot: E(|bias|) vs epi_std for both models."""
    if len(stats.diff_A) == 0 or stats.epi_var_A is None:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    bias_a = np.linalg.norm(stats.diff_A, axis=1)
    bias_b = np.linalg.norm(stats.diff_B, axis=1)
    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(edges)

    centers, ma, mb = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 10:
            continue
        centers.append((lo + hi) / 2)
        ma.append(bias_a[m].mean())
        mb.append(bias_b[m].mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(centers, ma, "b-o", markersize=4, label="E(|bias_DER| | u_epi)")
    ax.plot(centers, mb, "r-s", markersize=4, label="E(|bias_gs| | u_epi)")
    ax.set_xlabel("Epistemic std")
    ax.set_ylabel("Mean |bias|")
    ax.set_xscale("log")
    ax.set_title(f"{split}: bias magnitude vs epistemic uncertainty")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bias_mag_vs_epi.png"), dpi=150)
    plt.close()


def plot_signed_bias_vs_epi(stats: StatsAccumulator, split: str, out_dir: str | None = None):
    """Multi-line: E(bias_x/y/z) vs epi_std, DER solid, gs dashed."""
    if len(stats.diff_A) == 0 or stats.epi_var_A is None:
        return
    save_dir = out_dir or os.path.join(OUT_DIR, split)
    os.makedirs(save_dir, exist_ok=True)

    epi = stats.epi_var_A.flatten()
    da, db = stats.diff_A, stats.diff_B
    edges = np.percentile(epi, np.linspace(0, 100, 21))
    edges = np.unique(edges)

    centers, ax_a, ay_a, az_a, ax_b, ay_b, az_b = [], [], [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (epi >= lo) & (epi < hi)
        if m.sum() < 10:
            continue
        centers.append((lo + hi) / 2)
        ax_a.append(da[m, 0].mean())
        ay_a.append(da[m, 1].mean())
        az_a.append(da[m, 2].mean())
        ax_b.append(db[m, 0].mean())
        ay_b.append(db[m, 1].mean())
        az_b.append(db[m, 2].mean())

    colors = {"x": "C0", "y": "C1", "z": "C2"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for axis, ca, cb in [("x", ax_a, ax_b), ("y", ay_a, ay_b), ("z", az_a, az_b)]:
        c = colors[axis]
        ax.plot(centers, ca, "-o", color=c, markersize=3, label=f"DER bias_{axis}")
        ax.plot(centers, cb, "--s", color=c, markersize=3, alpha=0.6, label=f"gs  bias_{axis}")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Epistemic std")
    ax.set_ylabel("E(signed bias)")
    ax.set_xscale("log")
    ax.set_title(f"{split}: signed bias vs epistemic uncertainty (— DER, -- gs)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "signed_bias_vs_epi.png"), dpi=150)
    plt.close()
