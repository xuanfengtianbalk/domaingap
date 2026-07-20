"""alpha_eval.py — Evaluate DER pose error improvement via alpha correction on sunlamp/lightbox.

Loads the alpha calibration CSV, corrects coordinate predictions per axis,
runs PnP on original and corrected coords, computes angle/distance metrics.

Run: python alpha_eval.py --splits sunlamp lightbox --std_threshold 1.0
"""

from __future__ import annotations
import sys, os, json, csv, math, argparse, numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dinov3_main"))

from analysis_utils import load_model, build_dataloader
from utils_datasets.speedplus_utils_main.utils import Camera
from post_process import pose_calculats_from_coors, compute_pose_error, to_pnp_coors


# ── alpha table loader ───────────────────────────────────────────────────────

def load_alpha_table(csv_path: str) -> dict:
    """Read alpha_cali.csv → per-axis lookup dict.

    Returns: {axis: {"edges_ts": [float], "edges_pred": [float],
                     "grid": {(ts_bin, pred_bin): mean_alpha}}}
    """
    rows = list(csv.DictReader(open(csv_path)))
    tables = {}
    for key in ["x", "y", "z"]:
        ax_rows = [r for r in rows if r["axis"] == key]
        if not ax_rows:
            tables[key] = {"edges_ts": [], "edges_pred": [], "grid": {}}
            continue
        # edges from unique sorted values
        tlo = sorted(set(float(r["total_std_lo"]) for r in ax_rows))
        thi = sorted(set(float(r["total_std_hi"]) for r in ax_rows))
        plo = sorted(set(float(r["pred_lo"]) for r in ax_rows))
        phi = sorted(set(float(r["pred_hi"]) for r in ax_rows))
        # build combined ts edges: first lo + last hi
        ts_edges = tlo + [thi[-1]] if thi else tlo
        p_edges  = plo + [phi[-1]] if phi else plo
        # build grid
        grid = {}
        for r in ax_rows:
            ts_bin = int(r["total_std_bin"])
            p_bin  = int(r["pred_bin"])
            grid[(ts_bin, p_bin)] = float(r["mean_alpha"])
        tables[key] = {"edges_ts": np.array(ts_edges), "edges_pred": np.array(p_edges), "grid": grid}
    return tables


# ── coordinate correction ────────────────────────────────────────────────────

def correct_coords(coords_tensor: torch.Tensor, logl: torch.Tensor, loga: torch.Tensor,
                   logb: torch.Tensor, alpha_table: dict, std_min: float, std_max: float) -> tuple:
    """Apply per-axis alpha correction. Returns (coords_corrected, nan_mask).

    Pixels with any per-axis total_std outside [std_min, std_max] are set to NaN.
    Returns the NaN mask so caller can apply to original coords for fair comparison.
    """
    import numpy as np

    device = coords_tensor.device
    coords = coords_tensor.clone()
    B, C, H, W = coords.shape

    logl_np = logl.squeeze(0).cpu().numpy()  # (3, H, W)
    loga_np = loga.squeeze(0).cpu().numpy()
    logb_np = logb.squeeze(0).cpu().numpy()
    a = np.exp(loga_np) + 1.0 + 1e-6
    b = np.exp(logb_np) + 1e-6
    v = np.exp(logl_np) + 1e-6
    epi_var = b / ((a - 1 + 1e-12) * (v + 1e-12))
    alea_var = b / (a - 1 + 1e-12)
    total_std = np.sqrt(np.maximum(epi_var + alea_var, 0.0))  # (3, H, W)

    # too-low std → keep original DER, do NOT correct, do NOT NaN
    # in-range std → apply alpha correction
    # too-high std → NaN (exclude from PnP)
    low = np.ones((H, W), dtype=bool)
    mid = np.ones((H, W), dtype=bool)
    for ax in range(3):
        low &= (total_std[ax] < std_min)
        mid &= (total_std[ax] >= std_min) & (total_std[ax] <= std_max)

    coords_np = coords.squeeze(0).cpu().numpy()  # (3, H, W)
    nan_mask = ~(low | mid)
    coords_np[:, nan_mask] = float("nan")

    for ax_idx, key in enumerate(["x", "y", "z"]):
        tbl = alpha_table[key]
        ts_edges = tbl["edges_ts"]
        p_edges  = tbl["edges_pred"]
        grid     = tbl["grid"]
        if len(ts_edges) == 0 or len(p_edges) == 0:
            continue

        ts_ax = total_std[ax_idx]
        pr_ax = coords_np[ax_idx]

        for i, (tlo, thi) in enumerate(zip(ts_edges[:-1], ts_edges[1:])):
            for j, (plo, phi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
                ma = grid.get((i, j))
                if ma is None:
                    continue
                mask = (ts_ax >= tlo) & (ts_ax < thi) & (pr_ax >= plo) & (pr_ax < phi) & mid
                if mask.sum() == 0:
                    continue
                coords_np[ax_idx, mask] = pr_ax[mask] + ma * pr_ax[mask] * ts_ax[mask]

    result = torch.from_numpy(coords_np).unsqueeze(0).to(device).to(coords.dtype)
    return result, nan_mask


# ── PnP wrapper ──────────────────────────────────────────────────────────────

def run_pnp(outputs_raw: dict, gtbbox: torch.Tensor, qgt: torch.Tensor, rgt: torch.Tensor) -> tuple:
    """Run PnP on raw model outputs. Returns (angle_deg, dist_abs, is_true)."""
    activate = torch.nn.Sigmoid()
    coormap = outputs_raw["c"].clone().detach()
    mask_bool = (activate(outputs_raw["mask"]) > 0.5).expand_as(coormap).cpu()
    coormap[~mask_bool] = float("nan")
    coormap_np = to_pnp_coors(coormap.squeeze())

    gtb = gtbbox.cpu().detach().numpy()
    try:
        is_true, qvecs, tvecs = pose_calculats_from_coors(Camera.K, coormap_np, gtb[0])
    except Exception:
        is_true, qvecs, tvecs = False, None, None

    if not is_true:
        return float("nan"), float("nan"), False

    qg = qgt.squeeze().cpu()
    rg = rgt.squeeze().cpu()
    err_ori_deg, _, err_r_abs, _, _, _ = compute_pose_error(qvecs, tvecs, qg, rg, is_true)
    return float(err_ori_deg), float(err_r_abs), True


# ── main eval ────────────────────────────────────────────────────────────────

def evaluate(alpha_csv: str, splits: list, max_samples: int | None,
             std_min: float, std_max: float, device: str = "cuda:0", uuid: str | None = None):
    """Run alpha-corrected PnP evaluation on each split."""

    import yaml as _yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f)

    alpha_table = load_alpha_table(alpha_csv)
    print(f"Loaded alpha table: {sum(len(t['grid']) for t in alpha_table.values())} cells")

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "alpha_eval")
    os.makedirs(out_dir, exist_ok=True)

    if uuid is None:
        uuid = cfg["pairwise"]["uuid_1"]
    from analysis_utils import get_model_type_from_traininfo
    mt, bb = get_model_type_from_traininfo(uuid)
    model_type = mt[0]
    print(f"Model: {model_type} on {bb}")
    model, bc = load_model(uuid, model_type, bb, device)
    model.eval()

    for split in splits:
        print(f"\n=== {split} ===")
        dl = build_dataloader(uuid, split, max_samples=max_samples, batch_size=1)

        der_angles, der_dists = [], []
        corr_angles, corr_dists = [], []
        per_image = []

        n_total = 0
        for samples, targets in tqdm(dl, desc=split, ncols=80):
            image = samples.to(device)
            gtbbox = torch.round(targets["boxes"].squeeze(0))
            qgt = targets["q_gt"].squeeze()
            rgt = targets["r_gt"].squeeze()
            n_total += 1

            # clean forward
            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs_raw = model(image)

            # ── DER original PnP (unfiltered) ──
            angle_orig, dist_orig, ok_orig = run_pnp(outputs_raw, gtbbox, qgt, rgt)

            # ── DER corrected PnP (std filtered + alpha correction) ──
            coords_corr, std_mask = correct_coords(
                outputs_raw["c"].clone(), outputs_raw["logl"].clone(),
                outputs_raw["loga"].clone(), outputs_raw["logb"].clone(),
                alpha_table, std_min, std_max)

            # ── DER corrected PnP ──
            outputs_corr = dict(outputs_raw)
            outputs_corr["c"] = coords_corr
            angle_corr, dist_corr, ok_corr = run_pnp(outputs_corr, gtbbox, qgt, rgt)

            if ok_orig:
                der_angles.append(angle_orig)
                der_dists.append(dist_orig)
            if ok_corr:
                corr_angles.append(angle_corr)
                corr_dists.append(dist_corr)

            per_image.append({
                "angle_orig": angle_orig, "dist_orig": dist_orig,
                "angle_corr": angle_corr, "dist_corr": dist_corr,
            })

        # aggregate
        def stats(arr):
            if not arr:
                return {"mean": None, "std": None, "med": None, "n": 0}
            a = np.array(arr)
            return {"mean": float(a.mean()), "std": float(a.std()),
                    "med": float(np.median(a)), "n": len(a)}

        result = {
            "split":       split,
            "n_images":    n_total,
            "std_range": {"min": std_min, "max": std_max},
            "DER":       {"angle": stats(der_angles), "dist": stats(der_dists)},
            "CORRECTED": {"angle": stats(corr_angles), "dist": stats(corr_dists)},
            "per_image": per_image,
        }

        base = os.path.join(out_dir, split)
        with open(f"{base}.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(f"{base}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["angle_orig", "dist_orig", "angle_corr", "dist_corr"])
            w.writeheader()
            w.writerows(per_image)
        print(f"  → {base}.json / .csv")

        print(f"  DER:       angle={result['DER']['angle']['mean']:.2f}° ± {result['DER']['angle']['std']:.2f}  dist={result['DER']['dist']['mean']:.4f}  n={result['DER']['angle']['n']}")
        print(f"  CORRECTED: angle={result['CORRECTED']['angle']['mean']:.2f}° ± {result['CORRECTED']['angle']['std']:.2f}  dist={result['CORRECTED']['dist']['mean']:.4f}  n={result['CORRECTED']['angle']['n']}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha-corrected PnP evaluation on sunlamp/lightbox")
    parser.add_argument("--uuid", default=None,
                        help="Model UUID (default: pairwise.uuid_1 from config.yaml)")
    parser.add_argument("--alpha_csv", default=os.path.join(PROJECT_ROOT, "outputs", "alpha_cali", "alpha_cali.csv"))
    parser.add_argument("--splits", nargs="*", default=["sunlamp", "lightbox"])
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--std_min", type=float, default=0.0,
                        help="Per-axis total_std lower bound (pixels outside → NaN)")
    parser.add_argument("--std_max", type=float, default=10.0,
                        help="Per-axis total_std upper bound (pixels outside → NaN)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    evaluate(args.alpha_csv, args.splits, args.max_samples, args.std_min, args.std_max, args.device, args.uuid)
