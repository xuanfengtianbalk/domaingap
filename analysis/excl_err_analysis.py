"""excl_err_analysis.py — Per-image feature extraction + error-relationship analysis.

Relationships of interest (all stored as NUMBERS in CSV/JSON, no figures):
  * output error (baseline angle error b_i)  <->  2D/3D uncertainty, bbox, rot,
      model-mask fraction, predicted-coordinate centroid/spread
  * exclusion benefit delta_i               <->  the same features
  * control: b_i <-> delta_i

GT is used only to COMPUTE b_i/delta_i (analysis); the rule features
(uncertainty, bbox, rot, mask fraction, predicted-coord stats) are all
test-time available.

Outputs into outputs/excl_ablation_{uuid}/err_analysis/:
  per_image_{split}.csv  (one row per image, all features + b_i + delta_i)
  summary_{split}.json   (spearman rho + Fisher CI per feature vs b_i/delta_i,
                          plus 10-bin tables)
"""
import os
import sys
import json
import csv
import argparse
import multiprocessing as mp

import numpy as np
import torch
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from post_process import pose_calculats_from_coors, compute_pose_error, to_pnp_coors
from utils_datasets.speedplus_utils_main.utils import Camera

CONFIGS = [
    ("sunlamp", (0.10, 0.05, 0.10), (0.10, 0.10, 0.10)),
    ("lightbox", (0.00, 0.05, 0.10), (0.075, 0.075, 0.075)),
]

FEATURE_COLS = [
    "baseline_angle", "excl_angle", "delta",
    "mean_ts3d", "p50_ts3d", "p90_ts3d",
    "mean_ts2d", "p50_ts2d", "p90_ts2d",
    "bbox_area", "bbox_aspect", "rot_deg", "mask_frac",
    "mean_cx", "mean_cy", "mean_cz",
    "std_cx", "std_cy", "std_cz",
]

# feature -> (column, human name) used for relationship summaries
REL_FEATURES = [
    ("mean_ts3d", "mean_ts3d"),
    ("p50_ts3d", "p50_ts3d"),
    ("p90_ts3d", "p90_ts3d"),
    ("mean_ts2d", "mean_ts2d"),
    ("p50_ts2d", "p50_ts2d"),
    ("p90_ts2d", "p90_ts2d"),
    ("bbox_area", "bbox_area"),
    ("bbox_aspect", "bbox_aspect"),
    ("rot_deg", "rot_deg"),
    ("mask_frac", "mask_frac"),
    ("mean_cx", "mean_cx"),
    ("mean_cy", "mean_cy"),
    ("mean_cz", "mean_cz"),
    ("std_cx", "std_cx"),
    ("std_cy", "std_cy"),
    ("std_cz", "std_cz"),
]


def _ts_maps(logl, loga, logb):
    """Returns (ts3d, ts2d): per-pixel scalar uncertainties.
    ts3d = sqrt(ts_x^2+ts_y^2+ts_z^2), ts2d = sqrt(ts_x^2+ts_y^2)."""
    a = np.exp(loga) + 1.0 + 1e-6
    b = np.exp(logb) + 1e-6
    v = np.exp(logl) + 1e-6
    epi_var = b / ((a - 1 + 1e-12) * (v + 1e-12))
    alea_var = b / (a - 1 + 1e-12)
    ts_3d = np.sqrt(np.maximum(epi_var + alea_var, 0.0))  # (3,H,W)
    ts3d = np.sqrt((ts_3d ** 2).sum(axis=0))
    ts2d = np.sqrt((ts_3d[0] ** 2 + ts_3d[1] ** 2))
    return ts3d, ts2d


def _pnP_angle(c_np, K, gtb, q_gt, r_gt):
    try:
        is_true, qvecs, tvecs = pose_calculats_from_coors(
            K, to_pnp_coors(torch.from_numpy(c_np)), gtb)
    except Exception:
        return None
    if not is_true:
        return None
    err, _, _, _, _, _ = compute_pose_error(
        qvecs, tvecs, torch.from_numpy(q_gt), torch.from_numpy(r_gt), True)
    return float(err)


def _worker(job):
    split, center, radius, raw_dir, K = job
    sp_dir = os.path.join(raw_dir, split)
    names = sorted(os.listdir(sp_dir), key=lambda x: int(x.split("_")[1].split(".")[0]))
    cx, cy, cz = center
    rx, ry, rz = radius
    rows = []
    for fn in tqdm(names, desc=f"feat {split}", ncols=80):
        d = np.load(os.path.join(sp_dir, fn))
        c_np = d["c"].astype(np.float32)
        mask_bool = (torch.sigmoid(torch.from_numpy(d["mask"])) > 0.5).cpu().numpy()
        c_np[~np.broadcast_to(mask_bool, c_np.shape)] = float("nan")
        valid = np.isfinite(c_np[0])
        if valid.sum() < 20:
            continue

        b_i = _pnP_angle(c_np, K, d["boxes"], d["q_gt"], d["r_gt"])
        if b_i is None:
            continue

        zone = ((c_np[0] >= cx - rx) & (c_np[0] <= cx + rx)) | \
               ((c_np[1] >= cy - ry) & (c_np[1] <= cy + ry)) | \
               ((c_np[2] >= cz - rz) & (c_np[2] <= cz + rz))
        zone &= valid
        c_excl = c_np.copy()
        c_excl[:, zone] = float("nan")
        e_i = _pnP_angle(c_excl, K, d["boxes"], d["q_gt"], d["r_gt"])
        if e_i is None:
            continue

        ts3d, ts2d = _ts_maps(d["logl"], d["loga"], d["logb"])
        ts3d_v = ts3d[valid]
        ts2d_v = ts2d[valid]

        x1, y1, x2, y2 = d["boxes"]
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)

        c_valid = c_np[:, valid]
        mean_c = c_valid.mean(axis=1)
        std_c = c_valid.std(axis=1)

        rows.append([
            float(b_i), e_i, e_i - float(b_i),
            float(ts3d_v.mean()), float(np.percentile(ts3d_v, 50)), float(np.percentile(ts3d_v, 90)),
            float(ts2d_v.mean()), float(np.percentile(ts2d_v, 50)), float(np.percentile(ts2d_v, 90)),
            float(w * h), float(w / h),
            float(np.linalg.norm(d["r_gt"])) * 180.0 / np.pi,
            float(valid.mean()),
            float(mean_c[0]), float(mean_c[1]), float(mean_c[2]),
            float(std_c[0]), float(std_c[1]), float(std_c[2]),
        ])
    return split, rows


def _spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30:
        return None, None, None, int(len(x))
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rho = np.corrcoef(rx, ry)[0, 1]
    n = len(x)
    z = np.arctanh(max(min(rho, 1 - 1e-9), -1 + 1e-9))
    se = 1.0 / np.sqrt(n - 3)
    lo = np.tanh(z - 1.96 * se)
    hi = np.tanh(z + 1.96 * se)
    return float(rho), float(lo), float(hi), int(n)


def _bins_table(x, y, n_bins=10):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < n_bins * 2:
        return []
    order = np.argsort(x)
    edges = np.array_split(order, n_bins)
    out = []
    for e in edges:
        out.append({
            "x_min": float(x[e].min()), "x_max": float(x[e].max()),
            "n": int(len(e)), "y_mean": float(y[e].mean()),
            "y_std": float(y[e].std()) if len(e) > 1 else 0.0,
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", default="783632ef-ceaa-4499-9cf9-575d94303951")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    base = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", f"excl_ablation_{args.uuid}")
    raw_dir = os.path.join(base, "raw")
    out_dir = os.path.join(base, "err_analysis")
    os.makedirs(out_dir, exist_ok=True)
    K = Camera.K

    jobs = [(split, center, radius, raw_dir, K)
            for split, center, radius in CONFIGS]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        results = pool.map(_worker, jobs)

    for split, rows in results:
        csv_path = os.path.join(out_dir, f"per_image_{split}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx"] + FEATURE_COLS)
            for i, r in enumerate(rows):
                w.writerow([i] + r)

        col = {name: j for j, name in enumerate(FEATURE_COLS)}
        b = [r[col["baseline_angle"]] for r in rows]
        delta = [r[col["delta"]] for r in rows]
        summary = {
            "split": split, "n": len(rows),
            "baseline_angle_mean": float(np.mean(b)),
            "baseline_angle_std": float(np.std(b)),
            "delta_mean": float(np.mean(delta)),
            "delta_std": float(np.std(delta)),
            "correlations": {},
        }
        for fname, fname2 in REL_FEATURES:
            fv = [r[col[fname]] for r in rows]
            rho_b, lo_b, hi_b, n_b = _spearman(fv, b)
            rho_d, lo_d, hi_d, n_d = _spearman(fv, delta)
            summary["correlations"][fname] = {
                "vs_baseline_error": {"rho": rho_b, "ci95": [lo_b, hi_b], "n": n_b},
                "vs_delta": {"rho": rho_d, "ci95": [lo_d, hi_d], "n": n_d},
            }
        summary["bins_vs_baseline_error"] = {}
        summary["bins_vs_delta"] = {}
        for fname, fname2 in REL_FEATURES:
            fv = [r[col[fname]] for r in rows]
            summary["bins_vs_baseline_error"][fname] = _bins_table(fv, b)
            summary["bins_vs_delta"][fname] = _bins_table(fv, delta)
        with open(os.path.join(out_dir, f"summary_{split}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[err analysis] {split}: n={len(rows)} -> "
              f"{csv_path} + summary_{split}.json")

    print(f"err analysis outputs -> {out_dir}")


if __name__ == "__main__":
    main()
