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
