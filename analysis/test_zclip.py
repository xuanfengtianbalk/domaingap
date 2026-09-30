"""test_zclip.py — Offline validation: clip predicted z to the physical GT range
[0, 0.33] before PnP and compare against baseline (no network, reads saved npz).

Outputs: err_analysis/zclip_results.json (per split: baseline/clipped angle
mean±std, n) — numeric only, no figures.
"""
import os
import sys
import json
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

Z_MIN, Z_MAX = 0.0, 0.33  # GT physical z range (config.yaml gt_ranges)


def _pnP(c_np, K, gtb, q_gt, r_gt):
    try:
        ok, qv, tv = pose_calculats_from_coors(
            K, to_pnp_coors(torch.from_numpy(c_np)), gtb)
    except Exception:
        return None
    if not ok:
        return None
    err, _, _, _, _, _ = compute_pose_error(
        qv, tv, torch.from_numpy(q_gt), torch.from_numpy(r_gt), True)
    return float(err)


def _worker(job):
    split, raw_dir, K = job
    sp_dir = os.path.join(raw_dir, split)
    names = sorted(os.listdir(sp_dir), key=lambda x: int(x.split("_")[1].split(".")[0]))
    b_list, clip_list, both = [], [], []
    for fn in tqdm(names, desc=f"zclip {split}", ncols=80):
        d = np.load(os.path.join(sp_dir, fn))
        c = d["c"].astype(np.float32)
        mask_bool = (torch.sigmoid(torch.from_numpy(d["mask"])) > 0.5).cpu().numpy()
        c[~np.broadcast_to(mask_bool, c.shape)] = float("nan")
        b = _pnP(c, K, d["boxes"], d["q_gt"], d["r_gt"])
        cc = c.copy()
        cc[2] = np.clip(cc[2], Z_MIN, Z_MAX)
        e = _pnP(cc, K, d["boxes"], d["q_gt"], d["r_gt"])
        if b is not None:
            b_list.append(b)
        if e is not None:
            clip_list.append(e)
        if b is not None and e is not None:
            both.append((b, e))
    return split, b_list, clip_list, both


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", default="783632ef-ceaa-4499-9cf9-575d94303951")
    args = parser.parse_args()
    base = os.path.join(_ROOT, "outputs", f"excl_ablation_{args.uuid}")
    raw_dir = os.path.join(base, "raw")
    out_dir = os.path.join(base, "err_analysis")
    os.makedirs(out_dir, exist_ok=True)
    K = Camera.K

    jobs = [(split, raw_dir, K) for split in ("sunlamp", "lightbox")]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        results = pool.map(_worker, jobs)

    out = {}
    for split, b_list, clip_list, both in results:
        b = np.array(b_list); c = np.array(clip_list)
        out[split] = {
            "baseline_angle_mean": float(b.mean()),
            "baseline_angle_std": float(b.std()),
            "baseline_n": int(len(b)),
            "clipped_angle_mean": float(c.mean()),
            "clipped_angle_std": float(c.std()),
            "clipped_n": int(len(c)),
        }
        if both:
            bt = np.array([x[0] for x in both]); ct = np.array([x[1] for x in both])
            out[split]["paired_delta_mean"] = float((ct - bt).mean())
            out[split]["paired_n"] = int(len(both))
            out[split]["paired_n_benefit"] = int((ct < bt).sum())
            out[split]["paired_n_harm"] = int((ct > bt).sum())
        print(f"[zclip] {split}: baseline {b.mean():.3f}±{b.std():.2f} (n={len(b)}) "
              f"-> clipped {c.mean():.3f}±{c.std():.2f} (n={len(c)})")
        if both:
            bt = np.array([x[0] for x in both]); ct = np.array([x[1] for x in both])
            print(f"         paired delta {(ct-bt).mean():+.3f}, "
                  f"benefit {(ct<bt).mean()*100:.0f}% / harm {(ct>bt).mean()*100:.0f}%")
    with open(os.path.join(out_dir, "zclip_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[zclip] results -> {os.path.join(out_dir, 'zclip_results.json')}")


if __name__ == "__main__":
    main()
