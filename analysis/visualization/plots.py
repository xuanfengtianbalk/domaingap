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
