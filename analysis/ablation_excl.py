"""ablation_excl.py — Ablation study on exclusion parameters (center, radius, std, mode).

Evaluates 144 parameter combinations on sunlamp/lightbox using EXCLUDED mode only.
Output: ablation_{split}.csv with angle/dist stats (mean, std, p25, p50, p75).
"""

from __future__ import annotations
import sys, os, json, csv, math, itertools, argparse, numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dinov3_main"))

from analysis_utils import load_model, build_dataloader
from utils_datasets.speedplus_utils_main.utils import Camera
from post_process import pose_calculats_from_coors, compute_pose_error, to_pnp_coors


def _compute_excl_mask(c_np, center, radius_3, mode):
    if mode == "and":
        mask = np.ones(c_np.shape[1:], dtype=bool)
        for ax in range(3):
            mask &= (c_np[ax] >= center[ax] - radius_3[ax]) & \
                    (c_np[ax] <= center[ax] + radius_3[ax])
    else:
        mask = np.zeros(c_np.shape[1:], dtype=bool)
        for ax in range(3):
            mask |= (c_np[ax] >= center[ax] - radius_3[ax]) & \
                    (c_np[ax] <= center[ax] + radius_3[ax])
    return mask


def _total_std_scalar(logl, loga, logb):
    a = np.exp(loga) + 1.0 + 1e-6
    b = np.exp(logb) + 1e-6
    v = np.exp(logl) + 1e-6
    epi_var = b / ((a - 1 + 1e-12) * (v + 1e-12))
    alea_var = b / (a - 1 + 1e-12)
    ts_3d = np.sqrt(np.maximum(epi_var + alea_var, 0.0))
    return np.sqrt((ts_3d ** 2).sum(axis=0))


def run_pnp(coords_tensor, mask_tensor, gtbbox, qgt, rgt):
    activate = torch.nn.Sigmoid()
    cmap = coords_tensor.clone()
    mb = (activate(mask_tensor) > 0.5).expand_as(cmap).cpu()
    cmap[~mb] = float("nan")
    cmap_np = to_pnp_coors(cmap.squeeze())
    gtb = gtbbox.cpu().detach().numpy()
    if gtb.ndim == 2:
        gtb = gtb[0]
    try:
        is_true, qvecs, tvecs = pose_calculats_from_coors(Camera.K, cmap_np, gtb)
    except Exception:
        is_true, qvecs, tvecs = False, None, None
    if not is_true:
        return float("nan"), float("nan")
    qg = qgt.squeeze().cpu()
    rg = rgt.squeeze().cpu()
    err_ori_deg, _, err_r_abs, _, _, _ = compute_pose_error(qvecs, tvecs, qg, rg, is_true)
    return float(err_ori_deg), float(err_r_abs)


def main():
    parser = argparse.ArgumentParser(description="Ablation study on exclusion parameters")
    parser.add_argument("--uuid", default=None, help="Model UUID")
    parser.add_argument("--splits", nargs="*", default=["sunlamp", "lightbox"])
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    uuid = args.uuid or cfg["pairwise"]["uuid_1"]

    from analysis_utils import get_model_type_from_traininfo
    mt, bb = get_model_type_from_traininfo(uuid)
    model, bc = load_model(uuid, mt[0], bb, args.device)
    model.eval()

    # ── parameter grid ──
    centers = {
        "C1": (0.045, 0.057, 0.16),
        "C2": (0.0, 0.04, 0.165),
    }
    rxy_values = [0.025, 0.05, 0.1]
    rz_modes = ["uniform", "z_half"]
    std_values = [0.0025, 0.005, 0.01]
    excl_modes = ["and", "or"]

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "alpha_eval")
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits:
        print(f"\n=== {split} ===")
        dl = build_dataloader(uuid, split, max_samples=args.max_samples, batch_size=1)

        # accumulator: {combo_key: [angles], [dists]}
        acc_angles = defaultdict(list)
        acc_dists  = defaultdict(list)
        n_images = 0

        for samples, targets in tqdm(dl, desc=f"{split} ablation", ncols=80):
            image = samples.to(args.device)
            gtbbox = torch.round(targets["boxes"].squeeze())
            qgt = targets["q_gt"].squeeze()
            rgt = targets["r_gt"].squeeze()

            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs_raw = model(image)

            c_np = outputs_raw["c"].squeeze(0).cpu().numpy()  # (3, H, W)
            ts_scalar = _total_std_scalar(
                outputs_raw["logl"].squeeze(0).cpu().numpy(),
                outputs_raw["loga"].squeeze(0).cpu().numpy(),
                outputs_raw["logb"].squeeze(0).cpu().numpy())

            for c_name, center in centers.items():
                for rxy in rxy_values:
                    for rz_mode in rz_modes:
                        rz = rxy if rz_mode == "uniform" else rxy * 0.5
                        r3 = (rxy, rxy, rz)
                        for sm in std_values:
                            for mode in excl_modes:
                                mask = _compute_excl_mask(c_np, center, r3, mode)
                                if sm > 0:
                                    mask = mask & (ts_scalar > sm)
                                c_masked = c_np.copy()
                                c_masked[:, mask] = float("nan")
                                c_masked_t = torch.from_numpy(c_masked).unsqueeze(0).to(args.device).to(outputs_raw["c"].dtype)

                                outputs_masked = dict(outputs_raw)
                                outputs_masked["c"] = c_masked_t
                                angle, dist = run_pnp(c_masked_t, outputs_raw["mask"], gtbbox, qgt, rgt)

                                if not math.isnan(angle):
                                    key = (c_name, rxy, rz_mode, sm, mode)
                                    acc_angles[key].append(angle)
                                    acc_dists[key].append(dist)
            n_images += 1

        # ── write CSV ──
        csv_path = os.path.join(out_dir, f"ablation_{split}.csv")
        fieldnames = ["split", "excl_mode", "center", "cx", "cy", "cz",
                      "rz_mode", "rx", "ry", "rz", "std_excl_min",
                      "angle_mean", "angle_std", "angle_p25", "angle_p50", "angle_p75",
                      "dist_mean", "dist_std", "dist_p25", "dist_p50", "dist_p75",
                      "n_valid"]

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for c_name, rxy, rz_mode, sm, mode in itertools.product(
                    centers.keys(), rxy_values, rz_modes, std_values, excl_modes):
                key = (c_name, rxy, rz_mode, sm, mode)
                angles = np.array(acc_angles[key])
                dists  = np.array(acc_dists[key])
                n = len(angles)
                if n < 2:
                    continue
                center = centers[c_name]
                rz = rxy if rz_mode == "uniform" else rxy * 0.5
                w.writerow({
                    "split": split, "excl_mode": mode,
                    "center": c_name, "cx": center[0], "cy": center[1], "cz": center[2],
                    "rz_mode": rz_mode, "rx": rxy, "ry": rxy, "rz": rz,
                    "std_excl_min": sm,
                    "angle_mean": float(angles.mean()), "angle_std": float(angles.std()),
                    "angle_p25": float(np.percentile(angles, 25)),
                    "angle_p50": float(np.percentile(angles, 50)),
                    "angle_p75": float(np.percentile(angles, 75)),
                    "dist_mean": float(dists.mean()), "dist_std": float(dists.std()),
                    "dist_p25": float(np.percentile(dists, 25)),
                    "dist_p50": float(np.percentile(dists, 50)),
                    "dist_p75": float(np.percentile(dists, 75)),
                    "n_valid": n,
                })
        print(f"  → {csv_path} ({sum(1 for v in acc_angles.values() if len(v)>=2)} combos, {n_images} images)")


if __name__ == "__main__":
    main()
