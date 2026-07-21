"""pred_error_model.py — PCA + PLS analysis: model error from (total_std, pred_x/y/z).

Finds the relationship between predictive uncertainty, prediction position, and error.
PCA: dimensionality check.  PLS: regression model for error prediction.

Run: python pred_error_model.py --alpha_csv outputs/alpha_cali/alpha_cali.csv
"""

from __future__ import annotations
import sys, os, json, csv, argparse, warnings, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "pred_error_model")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(csv_path: str, max_samples: int | None = None) -> tuple:
    """Load per-cell data from alpha_cali.csv. Returns (X, y, axis_labels)."""
    rows = list(csv.DictReader(open(csv_path)))
    if max_samples and len(rows) > max_samples:
        # Weighted sampling by cell n
        weights = np.array([int(r["n"]) for r in rows], dtype=float)
        weights /= weights.sum()
        idx = np.random.choice(len(rows), max_samples, replace=False, p=weights)
        rows = [rows[i] for i in idx]

    features = []
    targets = []
    for r in rows:
        if int(r["n"]) < 5:
            continue
        features.append([
            float(r["mean_total_std"]),
            float(r["mean_pred"]),
        ])
        gt = float(r["mean_GT"])
        pred = float(r["mean_pred"])
        targets.append(abs(pred - gt))

    X = np.array(features)    # (N, 2): [total_std, pred]
    y = np.array(targets)     # (N,): |pred - GT|
    return X, y


def load_per_axis(csv_path: str, axis: str, max_samples: int | None = None) -> dict:
    """Load data for one axis. Returns {std, pred, error, X, y}."""
    rows = list(csv.DictReader(open(csv_path)))
    ax_rows = [r for r in rows if r["axis"] == axis and int(r["n"]) >= 5]
    if max_samples and len(ax_rows) > max_samples:
        weights = np.array([int(r["n"]) for r in ax_rows], dtype=float)
        weights /= weights.sum()
        idx = np.random.choice(len(ax_rows), max_samples, replace=False, p=weights)
        ax_rows = [ax_rows[i] for i in idx]

    stds = np.array([float(r["mean_total_std"]) for r in ax_rows])
    preds = np.array([float(r["mean_pred"]) for r in ax_rows])
    gts = np.array([float(r["mean_GT"]) for r in ax_rows])
    errors = np.abs(preds - gts)
    X = np.column_stack([stds, preds])
    return {"std": stds, "pred": preds, "error": errors, "X": X, "y": errors, "rows": ax_rows}


# ── PCA ──────────────────────────────────────────────────────────────────────

def run_pca(axis_data: dict, axis_name: str):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = axis_data["X"]
    y = axis_data["y"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # variance explained
    axes[0].bar(range(1, 3), pca.explained_variance_ratio_, color=["#1f77b4", "#ff7f0e"])
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["PC1", "PC2"])
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title(f"GT_{axis_name}: PCA variance")

    # loadings
    axes[1].barh(["total_std", f"pred_{axis_name}"], pca.components_[0], color="#1f77b4", alpha=0.7, label="PC1")
    axes[1].barh(["total_std", f"pred_{axis_name}"], pca.components_[1], color="#ff7f0e", alpha=0.5, label="PC2")
    axes[1].set_title("PCA loadings")
    axes[1].legend(fontsize=7)

    # error vs PC1
    axes[2].scatter(X_pca[:, 0], y, s=2, alpha=0.3, color="#1f77b4")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("|error|")
    axes[2].set_title("Error vs PC1")
    axes[2].grid(True, alpha=0.15)

    fig.suptitle(f"PCA analysis — GT_{axis_name}")
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"pca_{axis_name}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    result = {
        "axis": axis_name,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_loadings": {"total_std": float(pca.components_[0, 0]), f"pred_{axis_name}": float(pca.components_[0, 1])},
        "pc2_loadings": {"total_std": float(pca.components_[1, 0]), f"pred_{axis_name}": float(pca.components_[1, 1])},
    }
    print(f"  PCA {axis_name}: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"    PC1 loadings: std={result['pc1_loadings']['total_std']:.3f}, pred={result['pc1_loadings'][f'pred_{axis_name}']:.3f}")
    return result


# ── PLS ──────────────────────────────────────────────────────────────────────

def run_pls(axis_data: dict, axis_name: str, plot: bool = True):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler

    X = axis_data["X"]
    y = axis_data["y"]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    pls = PLSRegression(n_components=2)
    pls.fit(X_s, y_s)
    y_pred_s = pls.predict(X_s).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    r2 = float(1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rmse = float(np.sqrt(((y - y_pred) ** 2).mean()))
    mae  = float(np.abs(y - y_pred).mean())
    coef = pls.coef_.reshape(-1)

    coef = pls.coef_.reshape(-1)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(y, y_pred, s=2, alpha=0.3, color="#1f77b4")
        mx = max(y.max(), y_pred.max())
        axes[0].plot([0, mx], [0, mx], "k--", alpha=0.3)
        axes[0].set_xlabel("True |error|")
        axes[0].set_ylabel("Predicted |error|")
        axes[0].set_title(f"R2={r2:.3f} RMSE={rmse:.4f} MAE={mae:.4f}")
        axes[0].grid(True, alpha=0.15)

        imp = np.abs(coef)
        axes[1].bar(["total_std", f"pred_{axis_name}"], imp, color=["#1f77b4", "#ff7f0e"])
        axes[1].set_ylabel("|coefficient|")
        axes[1].set_title("PLS feature importance")

        fig.suptitle(f"PLS regression — GT_{axis_name}")
        plt.tight_layout()
        save_path = os.path.join(OUT_DIR, f"pls_{axis_name}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"  PLS {axis_name}: R2={r2:.3f} RMSE={rmse:.4f} MAE={mae:.4f}")
    result = {
        "axis": axis_name,
        "n_samples": len(y),
        "r2": r2, "rmse": rmse, "mae": mae,
        "coefficients": {"total_std": float(coef[0]), f"pred_{axis_name}": float(coef[1]) if len(coef) > 1 else 0.0},
    }
    print(f"  PLS {axis_name}: R2={r2:.3f} RMSE={rmse:.4f} MAE={mae:.4f}")
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PCA + PLS analysis: error ~ f(std, pred)")
    parser.add_argument("--alpha_csv", default=os.path.join(PROJECT_ROOT, "outputs", "alpha_cali", "alpha_cali.csv"))
    parser.add_argument("--max_samples", type=int, default=200000)
    parser.add_argument("--excl_cx", type=float, default=0.045, help="Exclusion zone center X")
    parser.add_argument("--excl_cy", type=float, default=0.057, help="Exclusion zone center Y")
    parser.add_argument("--excl_cz", type=float, default=0.16, help="Exclusion zone center Z")
    parser.add_argument("--sweep_r", nargs="*", type=float,
                        default=[0.10, 0.15, 0.20, 0.25, 0.30],
                        help="Exclusion radius sweep values")
    parser.add_argument("--skip_sweep", action="store_true", help="Skip exclusion sweep")
    args = parser.parse_args()

    center_map = {"x": args.excl_cx, "y": args.excl_cy, "z": args.excl_cz}
    sweep_r = args.sweep_r

    print(f"Loading {args.alpha_csv} ...")
    pca_results = {}
    pls_results = {}
    for axis_name in ["x", "y", "z"]:
        data = load_per_axis(args.alpha_csv, axis_name, args.max_samples)
        if len(data["y"]) < 10:
            print(f"  {axis_name}: insufficient data, skip")
            continue
        pca_results[axis_name] = run_pca(data, axis_name)
        pls_results[axis_name] = run_pls(data, axis_name)

    # ── exclusion sweep ──
    if not args.skip_sweep:
        sweep_rows = []
        for axis_name in ["x", "y", "z"]:
            data = load_per_axis(args.alpha_csv, axis_name, args.max_samples)
            center = center_map[axis_name]
            for r in sweep_r:
                keep = np.abs(data["pred"] - center) >= r
                if keep.sum() < 10:
                    continue
                X_f = data["X"][keep]
                y_f = data["y"][keep]
                filt_data = {"X": X_f, "y": y_f, "pred": data["pred"][keep],
                             "std": data["std"][keep], "error": y_f, "rows": None}
                res = run_pls(filt_data, f"{axis_name}_r{r:.2f}", plot=False)
                sweep_rows.append({
                    "axis": axis_name, "radius": r,
                    "r2": res["r2"], "rmse": res["rmse"], "mae": res["mae"],
                    "coef_std": res["coefficients"].get("total_std", 0),
                    "coef_pred": res["coefficients"].get(f"pred_{axis_name}", 0),
                })
                pct_excluded = float(1 - keep.mean()) * 100
                print(f"  sweep {axis_name} r={r:.2f}: R2={res['r2']:.3f} rmse={res['rmse']:.4f} excluded={pct_excluded:.1f}%")

        if sweep_rows:
            _plot_sweep_pls(sweep_rows)
            with open(os.path.join(OUT_DIR, "pls_sweep.json"), "w") as f:
                json.dump({"metadata": {"alpha_csv": args.alpha_csv, "sweep_r": sweep_r},
                           "rows": sweep_rows}, f, indent=2)

    # save
    meta = {"alpha_csv": args.alpha_csv, "max_samples": args.max_samples}
    for fname, data in [("pca", pca_results), ("pls", pls_results)]:
        path = os.path.join(OUT_DIR, f"{fname}_analysis.json")
        with open(path, "w") as f:
            json.dump({"metadata": meta, "results": data}, f, indent=2)
        print(f"  -> {path}")


def _plot_sweep_pls(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}

    for ax, metric, ylabel in [
        (axes[0], "r2", "R²"),
        (axes[1], "coef_std", "PLS coeff (total_std)"),
        (axes[2], "coef_pred", "PLS coeff (pred)"),
    ]:
        for axis_name in ["x", "y", "z"]:
            pts = [r for r in rows if r["axis"] == axis_name]
            pts.sort(key=lambda r: r["radius"])
            rr = [r["radius"] for r in pts]
            vv = [r[metric] for r in pts]
            ax.plot(rr, vv, "o-", color=colors[axis_name], markersize=5, linewidth=1.2, label=axis_name)
        ax.set_xlabel("Exclusion radius")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("PLS performance vs exclusion radius")
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "pls_sweep.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  -> {save_path}")


if __name__ == "__main__":
    main()
