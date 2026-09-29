"""excl_diagnose.py — Diagnose the best exclusion configs using the saved raw npz.

No network forward: reads outputs/excl_ablation_{uuid}/raw/*.npz + baseline npz.
Produces, per (config, split):
  1. ts distribution of excluded vs kept pixels
  2. spatial (image-position) map of excluded pixels
  3. per-image delta histogram (excluded - baseline angle)
  4. benefit/harm quartile features (rotation angle, bbox area) + correlations
  5. top-10 worst/best images
Plus center-sweep error-surface heatmaps (per cz slice, per split).
Outputs into outputs/excl_ablation_{uuid}/diagnose/.
"""
import os
import json
import csv
import argparse
import multiprocessing as mp

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from analysis_utils import get_model_type_from_traininfo  # noqa: F401 (import path)
from post_process import pose_calculats_from_coors, compute_pose_error, to_pnp_coors
from utils_datasets.speedplus_utils_main.utils import Camera

# best pure-geometry exclusion configs (from the radius/center ablation)
CONFIGS = [
    ("sunlamp_best", "sunlamp", (0.10, 0.05, 0.10), (0.10, 0.10, 0.10)),
    ("lightbox_best", "lightbox", (0.00, 0.05, 0.10), (0.075, 0.075, 0.075)),
]


def _ts_scalar(logl, loga, logb):
    a = np.exp(loga) + 1.0 + 1e-6
    b = np.exp(logb) + 1e-6
    v = np.exp(logl) + 1e-6
    epi_var = b / ((a - 1 + 1e-12) * (v + 1e-12))
    alea_var = b / (a - 1 + 1e-12)
    ts_3d = np.sqrt(np.maximum(epi_var + alea_var, 0.0))
    return np.sqrt((ts_3d ** 2).sum(axis=0))


def _worker(job):
    name, split, center, radius, raw_dir, baseline_path, K = job
    sp_dir = os.path.join(raw_dir, split)
    names = sorted(os.listdir(sp_dir), key=lambda x: int(x.split("_")[1].split(".")[0]))
    b = np.load(baseline_path)
    b_full = b["angles_full"]
    cx, cy, cz = center
    rx, ry, rz = radius

    deltas, recs = [], []
    ts_excl_all, ts_kept_all = [], []
    pos_u_all, pos_v_all = [], []
    angles_excl = []
    for idx, fn in enumerate(tqdm(names, desc=f"{name}", ncols=80)):
        d = np.load(os.path.join(sp_dir, fn))
        c_np = d["c"]
        mask_bool = (torch.sigmoid(torch.from_numpy(d["mask"])) > 0.5).cpu().numpy()
        c_np[~np.broadcast_to(mask_bool, c_np.shape)] = float("nan")
        # zone mask (pure geometric, or-mode)
        zone = (c_np[0] >= cx - rx) & (c_np[0] <= cx + rx)
        zone |= (c_np[1] >= cy - ry) & (c_np[1] <= cy + ry)
        zone |= (c_np[2] >= cz - rz) & (c_np[2] <= cz + rz)
        zone &= np.isfinite(c_np[0])
        kept = np.isfinite(c_np[0])
        m_frac = 1.0 - kept.mean()
        z_frac = float(zone.sum()) / float(kept.sum()) if kept.sum() > 0 else 0.0

        ts = _ts_scalar(d["logl"], d["loga"], d["logb"])
        if zone.sum() > 0:
            ts_excl_all.append(ts[zone])
        kept_not_zone = kept & (~zone)
        if kept_not_zone.sum() > 0:
            s = np.random.choice(int(kept_not_zone.sum()),
                                 min(5000, int(kept_not_zone.sum())), replace=False)
            ts_kept_all.append(ts[kept_not_zone][s])
        if zone.sum() > 0:
            idxs = np.argwhere(zone)
            s = np.random.choice(len(idxs), min(3000, len(idxs)), replace=False)
            H, W = c_np.shape[1], c_np.shape[2]
            pos_u_all.append(idxs[s, 1] / W)
            pos_v_all.append(idxs[s, 0] / H)

        c_excl = c_np.copy()
        c_excl[:, zone] = float("nan")
        try:
            is_true, qvecs, tvecs = pose_calculats_from_coors(
                K, to_pnp_coors(torch.from_numpy(c_excl)), d["boxes"])
        except Exception:
            is_true, qvecs, tvecs = False, None, None
        if not is_true or idx >= len(b_full) or not np.isfinite(b_full[idx]):
            continue
        err, _, dist, _, _, _ = compute_pose_error(
            qvecs, tvecs, torch.from_numpy(d["q_gt"]), torch.from_numpy(d["r_gt"]), True)
        err = float(err)
        delta = err - float(b_full[idx])
        deltas.append(delta)
        angles_excl.append(err)
        rot_deg = float(np.linalg.norm(d["r_gt"])) * 180.0 / np.pi
        x1, y1, x2, y2 = d["boxes"]
        area = float((x2 - x1) * (y2 - y1))
        recs.append((delta, rot_deg, area, m_frac, z_frac))

    out = {
        "name": name, "split": split,
        "deltas": deltas, "angles_excl": angles_excl, "recs": recs,
        "ts_excl": np.concatenate(ts_excl_all) if ts_excl_all else np.zeros(0),
        "ts_kept": np.concatenate(ts_kept_all) if ts_kept_all else np.zeros(0),
        "pos_u": np.concatenate(pos_u_all) if pos_u_all else np.zeros(0),
        "pos_v": np.concatenate(pos_v_all) if pos_v_all else np.zeros(0),
    }
    return out


def _plot_and_report(out, out_dir):
    name, split = out["name"], out["split"]
    d = np.array(out["deltas"])
    recs = np.array(out["recs"]) if out["recs"] else np.zeros((0, 5))
    summary = {"name": name, "split": split, "n": int(len(d))}
    if len(d):
        summary["delta_mean"] = float(d.mean())
        summary["delta_std"] = float(d.std())
        summary["delta_p25_p50_p75"] = [float(np.percentile(d, q)) for q in (25, 50, 75)]
        summary["n_benefit"] = int((d < 0).sum())
        summary["n_harm"] = int((d > 0).sum())
        # quartile features
        qs = np.percentile(d, [25, 50, 75])
        for qi, (lo, hi) in enumerate([(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]):
            m = (d >= lo) & (d <= hi)
            if m.sum():
                summary[f"q{qi+1}_n"] = int(m.sum())
                summary[f"q{qi+1}_delta"] = float(d[m].mean())
                summary[f"q{qi+1}_rot_deg"] = float(recs[m, 1].mean())
                summary[f"q{qi+1}_bbox_area"] = float(recs[m, 2].mean())
        if len(d) > 2:
            for feat, fn in [("rot_deg", 1), ("bbox_area", 2)]:
                summary[f"corr_delta_{feat}"] = float(np.corrcoef(d, recs[:, fn])[0, 1])
        order = np.argsort(d)
        summary["worst10"] = [(float(d[i]), float(recs[i, 1]), float(recs[i, 2])) for i in order[:10]]
        summary["best10"] = [(float(d[i]), float(recs[i, 1]), float(recs[i, 2])) for i in order[-10:][::-1]]

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.hist(d, bins=60)
        ax.axvline(d.mean(), color="r", linestyle="--", label=f"mean {d.mean():+.3f}")
        ax.set_title(f"{name}: per-image delta (excl - baseline), n={len(d)}")
        ax.set_xlabel("angle delta (deg)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_delta_hist.png"), dpi=150)
        plt.close(fig)

    if len(out["ts_excl"]) and len(out["ts_kept"]):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        bins = np.linspace(0, np.percentile(out["ts_kept"], 99.5), 60)
        ax.hist(out["ts_kept"], bins=bins, alpha=0.5, label="kept (out of zone)")
        ax.hist(out["ts_excl"], bins=bins, alpha=0.5, label="excluded (in zone)")
        ax.set_title(f"{name}: total-std distribution excluded vs kept")
        ax.set_xlabel("ts scalar")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_ts_hist.png"), dpi=150)
        plt.close(fig)

    if len(out["pos_u"]):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.hist2d(out["pos_u"], out["pos_v"], bins=40, range=[[0, 1], [0, 1]])
        ax.set_title(f"{name}: excluded-pixel image positions")
        ax.set_xlabel("u (norm)")
        ax.set_ylabel("v (norm)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_pos_map.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(out_dir, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _heatmaps(csv_path, out_dir):
    if not os.path.exists(csv_path):
        return
    rows = list(csv.DictReader(open(csv_path)))
    for split in ("sunlamp", "lightbox"):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        # center name format: c{cx:g}_{cy:g}_{cz:g}
        def parse_center(r):
            parts = r["center"].split("_")
            return float(parts[0][1:]), float(parts[1]), float(parts[2])
        czs = sorted({parse_center(r)[2] for r in rs})
        for cz in czs:
            grid = {}
            for r in rs:
                cx, cy, cz_r = parse_center(r)
                if cz_r != cz:
                    continue
                grid[(cx, cy)] = float(r["delta_angle_mean"])
            cxs = sorted({k[0] for k in grid})
            cys = sorted({k[1] for k in grid})
            M = np.zeros((len(cys), len(cxs)))
            for j, cy in enumerate(cys):
                for i, cx in enumerate(cxs):
                    M[j, i] = grid.get((cx, cy), np.nan)
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(M, cmap="RdYlGn_r", origin="lower")
            ax.set_xticks(range(len(cxs)))
            ax.set_xticklabels([f"{v:g}" for v in cxs])
            ax.set_yticks(range(len(cys)))
            ax.set_yticklabels([f"{v:g}" for v in cys])
            ax.set_title(f"{split} cz={cz}: delta angle")
            ax.set_xlabel("cx")
            ax.set_ylabel("cy")
            for j in range(len(cys)):
                for i in range(len(cxs)):
                    if np.isfinite(M[j, i]):
                        ax.text(i, j, f"{M[j, i]:+.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"heatmap_{split}_cz{cz}.png"), dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", default="783632ef-ceaa-4499-9cf9-575d94303951")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    base = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", f"excl_ablation_{args.uuid}")
    raw_dir = os.path.join(base, "raw")
    out_dir = os.path.join(base, "diagnose")
    os.makedirs(out_dir, exist_ok=True)
    K = Camera.K

    jobs = []
    for name, split, center, radius in CONFIGS:
        jobs.append((name, split, center, radius, raw_dir,
                     os.path.join(base, f"baseline_{split}.npz"), K))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        outs = pool.map(_worker, jobs)

    for out in outs:
        s = _plot_and_report(out, out_dir)
        print(json.dumps(s, indent=2))

    _heatmaps(os.path.join(base, "excl_center_sweep_results.csv"), out_dir)
    print(f"diagnose outputs -> {out_dir}")


if __name__ == "__main__":
    main()
