"""pred_error_model.py — PCA + PLS: error ~ f(total_std, pred_x, pred_y, pred_z).

Two modes:
  1. --save_npz        collect raw per-pixel 3D data from validation
  2. --raw_npz FILE    4D PLS + 3D exclusion sweep on saved pixels
  3. --alpha_csv FILE  legacy per-axis analysis from CSV (fallback)
"""

from __future__ import annotations
import sys, os, json, csv, argparse, warnings, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dinov3_main"))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "pred_error_model")
os.makedirs(OUT_DIR, exist_ok=True)


# ── raw pixel collection ──────────────────────────────────────────────────────

def collect_raw_pixels(uuid: str, max_samples: int, device: str, save_path: str):
    """Load DER model, run validation inference, save per-pixel 3D data."""
    import torch
    from analysis_utils import load_model, build_dataloader, get_model_type_from_traininfo
    from tqdm import tqdm

    mt, bb = get_model_type_from_traininfo(uuid)
    model, bc = load_model(uuid, mt[0], bb, device)
    model.eval()

    dl = build_dataloader(uuid, "validation", max_samples=max_samples, batch_size=1)

    MAX_TOTAL_PIX = 500_000
    all_pred, all_gt, all_ts, all_epi, all_alea = [], [], [], [], []
    n_collected = 0

    for samples, targets in tqdm(dl, desc="Collecting pixels", ncols=80):
        if n_collected >= MAX_TOTAL_PIX:
            break
        image = samples.to(device)
        mask = targets["mask_gt"][0] > 0.5
        gt = targets["coors_gt"][0].numpy()  # (3, H, W)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(image)

        pred = outputs["c"].squeeze(0).cpu().numpy()  # (3, H, W)
        logl = outputs["logl"].squeeze(0).cpu().numpy()
        loga = outputs["loga"].squeeze(0).cpu().numpy()
        logb = outputs["logb"].squeeze(0).cpu().numpy()
        a = np.exp(loga) + 1.0 + 1e-6
        b = np.exp(logb) + 1e-6
        v = np.exp(logl) + 1e-6
        epi_var = b / ((a - 1 + 1e-12) * (v + 1e-12))
        alea_var = b / (a - 1 + 1e-12)
        ts_total = np.sqrt(np.maximum(epi_var + alea_var, 0.0))
        ts_epi   = np.sqrt(np.maximum(epi_var, 0.0))
        ts_alea  = np.sqrt(np.maximum(alea_var, 0.0))

        valid_idx = np.where(mask.numpy().ravel())[0]
        if len(valid_idx) == 0:
            continue
        n_take = min(500, len(valid_idx))
        idx = np.random.choice(valid_idx, n_take, replace=False)

        for ax in range(3):
            flat = pred[ax].ravel()
            all_pred.append(flat[idx])
            all_gt.append(gt[ax].ravel()[idx])
            all_ts.append(ts_total[ax].ravel()[idx])
            all_epi.append(ts_epi[ax].ravel()[idx])
            all_alea.append(ts_alea[ax].ravel()[idx])
        n_collected += n_take

    pred_3d = np.column_stack([np.concatenate(all_pred[0::3]),
                                np.concatenate(all_pred[1::3]),
                                np.concatenate(all_pred[2::3])]).T  # (3, N)
    gt_3d   = np.column_stack([np.concatenate(all_gt[0::3]),
                                np.concatenate(all_gt[1::3]),
                                np.concatenate(all_gt[2::3])]).T
    ts_3d   = np.column_stack([np.concatenate(all_ts[0::3]),
                                np.concatenate(all_ts[1::3]),
                                np.concatenate(all_ts[2::3])]).T
    epi_3d  = np.column_stack([np.concatenate(all_epi[0::3]),
                                np.concatenate(all_epi[1::3]),
                                np.concatenate(all_epi[2::3])]).T
    alea_3d = np.column_stack([np.concatenate(all_alea[0::3]),
                                np.concatenate(all_alea[1::3]),
                                np.concatenate(all_alea[2::3])]).T

    np.savez(save_path, pred=pred_3d, gt=gt_3d, ts=ts_3d, epi=epi_3d, alea=alea_3d)
    print(f"  saved {pred_3d.shape[1]} pixels → {save_path}")


# ── raw data loader ───────────────────────────────────────────────────────────

def load_raw_data(npz_path: str, max_samples: int | None = None) -> dict:
    """Load raw pixel data. Returns dict with pred(N,3), gt(N,3), ts(N,3), etc."""
    d = np.load(npz_path)
    keys = list(d.keys())
    N = d["pred"].shape[1]  # (3, N)
    if max_samples and N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
    else:
        idx = slice(None)

    data = {
        "pred": d["pred"][:, idx].T,     # (N, 3)
        "gt":   d["gt"][:, idx].T,
        "ts":   d["ts"][:, idx].T,
    }
    if "epi" in keys:
        data["epi"] = d["epi"][:, idx].T
        data["alea"] = d["alea"][:, idx].T
    data["error_l2"] = np.linalg.norm(data["pred"] - data["gt"], axis=1)
    data["ts_scalar"] = np.sqrt((data["ts"] ** 2).sum(axis=1))
    return data


# ── 4D PLS ────────────────────────────────────────────────────────────────────

def run_pls_4d(data: dict, sweep_r: float | None = None,
               excl_center: tuple = None, plot: bool = True, label: str = ""):
    from sklearn.cross_decomposition import PLSRegression

    pred = data["pred"]
    error = data["error_l2"]
    ts_s  = data["ts_scalar"]
    N = len(error)

    if sweep_r is not None and excl_center is not None:
        keep = np.ones(N, dtype=bool)
        for ax in range(3):
            keep &= np.abs(pred[:, ax] - excl_center[ax]) >= sweep_r
        pred = pred[keep]
        error = error[keep]
        ts_s = ts_s[keep]
        pct_excluded = float(1 - keep.mean()) * 100
    else:
        pct_excluded = 0.0

    if len(error) < 10:
        return None

    X = np.column_stack([ts_s, pred[:, 0], pred[:, 1], pred[:, 2]])
    y = error

    pls = PLSRegression(n_components=2)
    pls.fit(X, y)
    y_pred = pls.predict(X).ravel()

    r2 = float(1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rmse = float(np.sqrt(((y - y_pred) ** 2).mean()))
    mae  = float(np.abs(y - y_pred).mean())
    coef = pls.coef_.reshape(-1)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(y, y_pred, s=2, alpha=0.3, color="#1f77b4")
        mx = max(y.max(), y_pred.max())
        axes[0].plot([0, mx], [0, mx], "k--", alpha=0.3)
        axes[0].set_xlabel("True L2 error")
        axes[0].set_ylabel("Predicted L2 error")
        axes[0].set_title(f"R²={r2:.3f} RMSE={rmse:.4f} MAE={mae:.4f}")
        axes[0].grid(True, alpha=0.15)

        names = ["total_std", "pred_x", "pred_y", "pred_z"]
        axes[1].bar(names, np.abs(coef), color=["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"])
        axes[1].set_ylabel("|coefficient|")
        axes[1].set_title("PLS feature importance")
        plt.xticks(rotation=30, fontsize=8)

        title = "PLS regression (4D)" + (f" — {label}" if label else "")
        fig.suptitle(title)
        plt.tight_layout()
        save_path = os.path.join(OUT_DIR, f"pls_4d{'_'+label if label else ''}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

    result = {
        "r2": r2, "rmse": rmse, "mae": mae, "n_samples": len(y),
        "pct_excluded": pct_excluded,
        "coeff": {n: float(coef[i]) for i, n in enumerate(["total_std", "pred_x", "pred_y", "pred_z"]) if i < len(coef)},
    }
    return result


# ── 4D PCA ────────────────────────────────────────────────────────────────────

def run_pca_4d(data: dict):
    from sklearn.decomposition import PCA
    pred = data["pred"]
    ts_s = data["ts_scalar"]
    X = np.column_stack([ts_s, pred[:, 0], pred[:, 1], pred[:, 2]])
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(range(1, 5), pca.explained_variance_ratio_, color=["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("PCA variance explained")

    names = ["total_std", "pred_x", "pred_y", "pred_z"]
    for ci in range(2):
        axes[1+ci].barh(names, pca.components_[ci], color=["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"])
        axes[1+ci].set_title(f"PC{ci+1} loadings")

    fig.suptitle("PCA analysis (4D)")
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "pca_4d.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1": {n: float(pca.components_[0, i]) for i, n in enumerate(names)},
        "pc2": {n: float(pca.components_[1, i]) for i, n in enumerate(names)},
    }


# ── 3D exclusion sweep ───────────────────────────────────────────────────────

def run_sweep_4d(data: dict, excl_center: tuple, sweep_r: list):
    rows = []
    for r in sweep_r:
        res = run_pls_4d(data, sweep_r=r, excl_center=excl_center, plot=False)
        if res:
            res["radius"] = r
            rows.append(res)
            print(f"  sweep r={r:.2f}: R²={res['r2']:.3f} rmse={res['rmse']:.4f} excluded={res['pct_excluded']:.1f}%")

    if not rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rr = [r["radius"] for r in rows]
    axes[0].plot(rr, [r["r2"] for r in rows], "bo-", markersize=6)
    axes[0].set_xlabel("Exclusion radius")
    axes[0].set_ylabel("R²")
    axes[0].set_title("PLS R² vs exclusion radius")
    axes[0].grid(True, alpha=0.2)

    names = ["total_std", "pred_x", "pred_y", "pred_z"]
    colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    for ni, name in enumerate(names):
        coefs = [np.abs(r["coeff"].get(name, 0)) for r in rows]
        axes[1].plot(rr, coefs, "o-", color=colors[ni], markersize=4, label=name)
    axes[1].set_xlabel("Exclusion radius")
    axes[1].set_ylabel("|coefficient|")
    axes[1].set_title("PLS coefficient vs exclusion radius")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle("PLS 4D: 3D exclusion sweep")
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "pls_sweep_4d.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    with open(os.path.join(OUT_DIR, "pls_sweep_4d.json"), "w") as f:
        json.dump({"sweep_r": sweep_r, "rows": rows}, f, indent=2)
    print(f"  -> {save_path}")


# ── per-axis PLS (from raw 3D data, axis-specific error) ────────────────────

def run_pls_per_axis(data: dict, axis_idx: int, axis_name: str,
                     excl_center: tuple = None, sweep_r: float = None, plot: bool = True):
    from sklearn.cross_decomposition import PLSRegression

    pred = data["pred"][:, axis_idx]
    gt   = data["gt"][:, axis_idx]
    ts   = data["ts"][:, axis_idx]
    error = np.abs(pred - gt)
    N = len(error)

    if sweep_r is not None and excl_center is not None:
        keep = np.abs(pred - excl_center[axis_idx]) >= sweep_r
        pred = pred[keep]; gt = gt[keep]; ts = ts[keep]; error = error[keep]
        pct_excluded = float(1 - keep.mean()) * 100
    else:
        pct_excluded = 0.0

    if len(error) < 10:
        return None

    X = np.column_stack([ts, pred])
    y = error

    pls = PLSRegression(n_components=2)
    pls.fit(X, y)
    y_pred = pls.predict(X).ravel()

    r2 = float(1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rmse = float(np.sqrt(((y - y_pred) ** 2).mean()))
    mae  = float(np.abs(y - y_pred).mean())
    coef = pls.coef_.reshape(-1)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(y, y_pred, s=2, alpha=0.3, color="#1f77b4")
        mx = max(y.max(), y_pred.max())
        axes[0].plot([0, mx], [0, mx], "k--", alpha=0.3)
        axes[0].set_xlabel(f"True |error_{axis_name}|")
        axes[0].set_ylabel(f"Predicted |error_{axis_name}|")
        axes[0].set_title(f"R²={r2:.3f} RMSE={rmse:.4f} MAE={mae:.4f}")
        axes[0].grid(True, alpha=0.15)

        axes[1].bar(["total_std", f"pred_{axis_name}"], np.abs(coef), color=["#d62728", "#1f77b4"])
        axes[1].set_ylabel("|coefficient|")
        axes[1].set_title("PLS feature importance")

        fig.suptitle(f"PLS regression — axis {axis_name}")
        plt.tight_layout()
        save_path = os.path.join(OUT_DIR, f"pls_raw_{axis_name}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

    return {
        "axis": axis_name, "r2": r2, "rmse": rmse, "mae": mae, "n_samples": len(y),
        "pct_excluded": pct_excluded,
        "coeff": {"total_std": float(coef[0]), f"pred_{axis_name}": float(coef[1]) if len(coef) > 1 else 0.0},
    }


def run_sweep_per_axis(data: dict, excl_center: tuple, sweep_r: list):
    colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax_idx, ax_name in enumerate(["x", "y", "z"]):
        rows = []
        for r in sweep_r:
            res = run_pls_per_axis(data, ax_idx, ax_name, excl_center, sweep_r=r, plot=False)
            if res:
                res["radius"] = r
                rows.append(res)
                print(f"  sweep {ax_name} r={r:.2f}: R²={res['r2']:.3f} rmse={res['rmse']:.4f} excluded={res['pct_excluded']:.1f}%")

        if not rows:
            continue
        rr = [r["radius"] for r in rows]
        axes[0].plot(rr, [r["r2"] for r in rows], "o-", color=colors[ax_name], markersize=5, label=f"{ax_name} R²")
        axes[1].plot(rr, [np.abs(r["coeff"]["total_std"]) for r in rows], "o-", color=colors[ax_name], markersize=5, label=f"{ax_name} std")
        axes[2].plot(rr, [np.abs(r["coeff"][f"pred_{ax_name}"]) for r in rows], "o-", color=colors[ax_name], markersize=5, label=f"{ax_name} pred")

    for ax, ylabel, title in [
        (axes[0], "R²", "R² vs exclusion radius"),
        (axes[1], "|coeff total_std|", "total_std coefficient"),
        (axes[2], "|coeff pred|", "pred coefficient"),
    ]:
        ax.set_xlabel("Exclusion radius")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Per-axis PLS: exclusion sweep (raw pixel data)")
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "pls_sweep_raw_per_axis.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  -> {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Std vs error analysis with exclusion sweep")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--raw_npz", type=str, default=None)
    parser.add_argument("--uuid", default=None)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--excl_cx", type=float, default=0.045)
    parser.add_argument("--excl_cy", type=float, default=0.057)
    parser.add_argument("--excl_cz", type=float, default=0.16)
    parser.add_argument("--sweep_r", nargs="*", type=float,
                        default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.save_npz:
        import yaml as _yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(cfg_path) as f:
            cfg = _yaml.safe_load(f)
        uuid = args.uuid or cfg["pairwise"]["uuid_1"]
        save_path = os.path.join(OUT_DIR, "alpha_cali_pixels.npz")
        collect_raw_pixels(uuid, args.max_samples, args.device, save_path)
        return

    if not args.raw_npz:
        print("Specify --raw_npz or --save_npz")
        return

    npz_path = args.raw_npz if os.path.isabs(args.raw_npz) else os.path.join(PROJECT_ROOT, args.raw_npz)
    if not os.path.exists(npz_path):
        npz_path = os.path.join(OUT_DIR, os.path.basename(args.raw_npz))
    data = load_raw_data(npz_path)
    print(f"Loaded {data['pred'].shape[0]} pixels")

    center = {"x": args.excl_cx, "y": args.excl_cy, "z": args.excl_cz}
    sweep_r = args.sweep_r

    # ── per-axis: R²(std, error) vs exclusion radius ──
    rows_all = []
    for ax_name, ax_idx in [("x", 0), ("y", 1), ("z", 2)]:
        pred = data["pred"][:, ax_idx]
        gt   = data["gt"][:, ax_idx]
        ts   = data["ts"][:, ax_idx]
        error = np.abs(pred - gt)
        cx = center[ax_name]

        for r in sweep_r:
            keep = np.abs(pred - cx) >= r
            if keep.sum() < 50:
                continue
            # R² of total_std alone
            from sklearn.linear_model import LinearRegression
            m = LinearRegression().fit(ts[keep].reshape(-1, 1), error[keep])
            y_pred = m.predict(ts[keep].reshape(-1, 1))
            ss_res = ((error[keep] - y_pred) ** 2).sum()
            ss_tot = ((error[keep] - error[keep].mean()) ** 2).sum()
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
            pct_excluded = float(1 - keep.mean()) * 100
            rows_all.append({
                "axis": ax_name, "radius": r,
                "r2": r2,
                "n": int(keep.sum()), "pct_excluded": pct_excluded,
            })

    #     # ── plot R² vs radius ──
    colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ax_name in ["x", "y", "z"]:
        pts = [r for r in rows_all if r["axis"] == ax_name]
        pts.sort(key=lambda r: r["radius"])
        rr, r2 = [r["radius"] for r in pts], [r["r2"] for r in pts]
        ax.plot(rr, r2, "o-", color=colors[ax_name], markersize=5, label=ax_name)
    ax.set_xlabel("Exclusion radius"); ax.set_ylabel("R²")
    ax.set_title("R²(std→error) vs exclusion radius")
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "std_vs_error_sweep.png"), dpi=200)
    plt.close()

    # ── std vs error by distance-from-center interval ──
    intervals = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, np.inf)]
    n_intervals = len(intervals)

    fig, axes = plt.subplots(3, n_intervals, figsize=(4*n_intervals, 10), sharex="col")
    for ri, (ax_name, ax_idx) in enumerate([("x", 0), ("y", 1), ("z", 2)]):
        pred = data["pred"][:, ax_idx]
        gt   = data["gt"][:, ax_idx]
        ts   = data["ts"][:, ax_idx]
        error = np.abs(pred - gt)
        dist  = np.abs(pred - center[ax_name])

        for ci, (dlo, dhi) in enumerate(intervals):
            ax = axes[ri, ci]
            keep = (dist >= dlo) & (dist < dhi)
            n_draw = min(8000, keep.sum())
            idx = np.where(keep)[0]
            if len(idx) < 10:
                continue
            if len(idx) > n_draw:
                idx = np.random.choice(idx, n_draw, replace=False)
            ax.scatter(ts[idx], error[idx], s=1, alpha=0.3, color="#1f77b4", rasterized=True)
            if dhi == np.inf:
                label = f"d≥{dlo:.2f}"
            else:
                label = f"d∈[{dlo:.2f},{dhi:.2f})"
            ax.set_title(f"{ax_name} {label}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, alpha=0.15)
            ax.set_xlabel("total_std")
            if ci == 0:
                ax.set_ylabel(f"|error_{ax_name}|")

    fig.suptitle("std vs error by pred-to-center distance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "std_vs_error_intervals.png"), dpi=200)
    plt.close()

    with open(os.path.join(OUT_DIR, "std_vs_error_sweep.json"), "w") as f:
        json.dump({"sweep_r": sweep_r, "rows": rows_all}, f, indent=2)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
