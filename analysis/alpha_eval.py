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
        is_true, qvecs, tvecs = pose_calculats_from_coors(Camera.K, coormap_np, gtb)
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
             std_min: float, std_max: float, device: str = "cuda:0", uuid: str | None = None,
             excl_center: tuple = None, excl_radius: tuple = None, corr_excl: bool = False,
             excl_mode: str = "and",
             excl_sweep_r: list = None, std_excl_min: float = 0.0):
    """Run alpha-corrected PnP evaluation on each split.

    Args:
        excl_center: (cx, cy, cz) — exclusion zone center
        excl_radius: (rx, ry, rz) — exclusion zone half-width per axis
        corr_excl: apply exclusion to CORRECTED mode as well
        excl_mode: "and" (all axes in zone) or "or" (any axis in zone)
        excl_sweep_r: list of radii for sweep mode (enables ratio vs error analysis)
    """

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

    has_excl = excl_center is not None and excl_radius is not None
    do_sweep = excl_sweep_r is not None and len(excl_sweep_r) > 0 and excl_center is not None

    for split in splits:
        print(f"\n=== {split} ===")
        dl = build_dataloader(uuid, split, max_samples=max_samples, batch_size=1)

        if do_sweep:
            _run_sweep(model, dl, alpha_table, std_min, std_max, excl_center, excl_radius, excl_mode,
                       corr_excl, excl_sweep_r, std_excl_min, out_dir, split, device)
            continue

        base_angles, base_dists = [], []
        excl_angles, excl_dists = [], []   # DER + exclusion filter
        corr_angles, corr_dists = [], []    # alpha corrected
        per_image = []
        # per-pixel pred vs std collection (sampled)
        std_preds, std_ts, std_epi, std_alea = [], [], [], []

        has_excl = excl_center is not None and excl_radius is not None

        n_total = 0
        for samples, targets in tqdm(dl, desc=split, ncols=80):
            image = samples.to(device)
            gtbbox = torch.round(targets["boxes"].squeeze())  # (1,1,4) → (4,)
            qgt = targets["q_gt"].squeeze()
            rgt = targets["r_gt"].squeeze()
            n_total += 1

            # clean forward
            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs_raw = model(image)

            # ── baseline: raw DER, no filtering ──
            angle_base, dist_base, ok_base = run_pnp(outputs_raw, gtbbox, qgt, rgt)

            # ── exclusion filter (shared for EXCLUDED + optional CORRECTED) ──
            angle_excl = dist_excl = ok_excl = None
            if has_excl:
                excl_c = outputs_raw["c"].clone()
                c_np = excl_c.squeeze(0).cpu().numpy()
                if excl_mode == "and":
                    mask = np.ones(c_np.shape[1:], dtype=bool)
                    op = np.logical_and
                else:
                    mask = np.zeros(c_np.shape[1:], dtype=bool)
                    op = np.logical_or
                for ax in range(3):
                    mask = op(mask, (c_np[ax] >= excl_center[ax] - excl_radius[ax]) &
                                    (c_np[ax] <= excl_center[ax] + excl_radius[ax]))
                # only exclude center pixels with high uncertainty
                if std_excl_min > 0:
                    logl = outputs_raw["logl"].squeeze(0).cpu().numpy()
                    loga = outputs_raw["loga"].squeeze(0).cpu().numpy()
                    logb = outputs_raw["logb"].squeeze(0).cpu().numpy()
                    a_np = np.exp(loga) + 1.0 + 1e-6
                    b_np = np.exp(logb) + 1e-6
                    v_np = np.exp(logl) + 1e-6
                    epi_var = b_np / ((a_np - 1 + 1e-12) * (v_np + 1e-12))
                    alea_var = b_np / (a_np - 1 + 1e-12)
                    ts_3d = np.sqrt(np.maximum(epi_var + alea_var, 0.0))
                    ts_scalar = np.sqrt((ts_3d ** 2).sum(axis=0))
                    mask = mask & (ts_scalar > std_excl_min)
                c_np[:, mask] = float("nan")
                excl_c = torch.from_numpy(c_np).unsqueeze(0).to(device)
                outputs_excl = dict(outputs_raw)
                outputs_excl["c"] = excl_c
                angle_excl, dist_excl, ok_excl = run_pnp(outputs_excl, gtbbox, qgt, rgt)

            # ── corrected: optional exclusion → alpha correction ──
            if has_excl and corr_excl:
                c_for_alpha = torch.from_numpy(c_np).unsqueeze(0).to(device).to(outputs_raw["c"].dtype)
                raw_for_alpha = dict(outputs_raw)
                raw_for_alpha["c"] = c_for_alpha
            else:
                raw_for_alpha = outputs_raw

            coords_corr, _ = correct_coords(
                raw_for_alpha["c"].clone(), raw_for_alpha["logl"].clone(),
                raw_for_alpha["loga"].clone(), raw_for_alpha["logb"].clone(),
                alpha_table, std_min, std_max)

            outputs_corr = dict(outputs_raw)
            outputs_corr["c"] = coords_corr
            angle_corr, dist_corr, ok_corr = run_pnp(outputs_corr, gtbbox, qgt, rgt)

            if ok_base:
                base_angles.append(angle_base)
                base_dists.append(dist_base)
            if has_excl and ok_excl:
                excl_angles.append(angle_excl)
                excl_dists.append(dist_excl)
            if ok_corr:
                corr_angles.append(angle_corr)
                corr_dists.append(dist_corr)

            # sample pixels for pred vs std plot (max ~5000 per image)
            c_np = outputs_raw["c"].squeeze(0).cpu().numpy()  # (3, H, W)
            logl = outputs_raw["logl"].squeeze(0).cpu().numpy()
            loga = outputs_raw["loga"].squeeze(0).cpu().numpy()
            logb = outputs_raw["logb"].squeeze(0).cpu().numpy()
            a_np = np.exp(loga) + 1.0 + 1e-6
            b_np = np.exp(logb) + 1e-6
            v_np = np.exp(logl) + 1e-6
            epi_var = b_np / ((a_np - 1 + 1e-12) * (v_np + 1e-12))
            alea_var = b_np / (a_np - 1 + 1e-12)
            total_std = np.sqrt(np.maximum(epi_var + alea_var, 0.0))
            epi_std    = np.sqrt(np.maximum(epi_var, 0.0))
            alea_std   = np.sqrt(np.maximum(alea_var, 0.0))
            n_pix = c_np.shape[1] * c_np.shape[2]
            take = min(n_pix, 5000)
            idx = np.random.choice(n_pix, take, replace=False)
            for ax in range(3):
                std_preds.append(c_np[ax].ravel()[idx])
                std_ts.append(total_std[ax].ravel()[idx])
                std_epi.append(epi_std[ax].ravel()[idx])
                std_alea.append(alea_std[ax].ravel()[idx])

            per_image.append({
                "angle_base": angle_base, "dist_base": dist_base,
                "angle_excl": angle_excl, "dist_excl": dist_excl,
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
            "BASELINE":  {"angle": stats(base_angles), "dist": stats(base_dists)},
            "CORRECTED": {"angle": stats(corr_angles), "dist": stats(corr_dists)},
            "per_image": per_image,
        }
        if has_excl:
            result["EXCLUDED"] = {"angle": stats(excl_angles), "dist": stats(excl_dists)}

        base = os.path.join(out_dir, split)
        with open(f"{base}.json", "w") as f:
            json.dump(result, f, indent=2)
        csv_fields = ["angle_base", "dist_base", "angle_excl", "dist_excl", "angle_corr", "dist_corr"]
        with open(f"{base}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(per_image)
        print(f"  → {base}.json / .csv")

        print(f"  BASELINE:   angle={result['BASELINE']['angle']['mean']:.2f}° ± {result['BASELINE']['angle']['std']:.2f}  dist={result['BASELINE']['dist']['mean']:.4f}  n={result['BASELINE']['angle']['n']}")
        if has_excl:
            print(f"  EXCLUDED:   angle={result['EXCLUDED']['angle']['mean']:.2f}° ± {result['EXCLUDED']['angle']['std']:.2f}  dist={result['EXCLUDED']['dist']['mean']:.4f}  n={result['EXCLUDED']['angle']['n']}")
        print(f"  CORRECTED:  angle={result['CORRECTED']['angle']['mean']:.2f}° ± {result['CORRECTED']['angle']['std']:.2f}  dist={result['CORRECTED']['dist']['mean']:.4f}  n={result['CORRECTED']['angle']['n']}")

        # pred vs std plot
        if len(std_preds) > 0:
            _plot_pred_vs_std(std_preds, std_ts, std_epi, std_alea, out_dir, split)


# ── exclusion ratio sweep ─────────────────────────────────────────────────────

def _run_sweep(model, dl, alpha_table, std_min, std_max, excl_center, excl_radius, excl_mode,
               corr_excl, sweep_r, std_excl_min, out_dir, split, device):
    """Sweep over exclusion radii + std threshold — PnP runs once per image, only ratio varies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    n_images = 0
    for samples, targets in tqdm(dl, desc=f"{split} sweep", ncols=80):
        image = samples.to(device)
        gtbbox = torch.round(targets["boxes"].squeeze())
        qgt = targets["q_gt"].squeeze()
        rgt = targets["r_gt"].squeeze()

        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs_raw = model(image)

        c_np_full = outputs_raw["c"].squeeze(0).cpu().numpy()  # (3, H, W)
        total_valid = np.isfinite(c_np_full[0]).sum()

        # scalar total_std per pixel
        logl = outputs_raw["logl"].squeeze(0).cpu().numpy()
        loga = outputs_raw["loga"].squeeze(0).cpu().numpy()
        logb = outputs_raw["logb"].squeeze(0).cpu().numpy()
        a_np = np.exp(loga) + 1.0 + 1e-6
        b_np = np.exp(logb) + 1e-6
        v_np = np.exp(logl) + 1e-6
        epi_var = b_np / ((a_np - 1 + 1e-12) * (v_np + 1e-12))
        alea_var = b_np / (a_np - 1 + 1e-12)
        ts_3d = np.sqrt(np.maximum(epi_var + alea_var, 0.0))  # (3, H, W)
        ts_scalar = np.sqrt((ts_3d ** 2).sum(axis=0))          # (H, W) scalar

        # ── BASELINE ──
        angle_base, dist_base, _ = run_pnp(outputs_raw, gtbbox, qgt, rgt)

        # ── EXCLUDED (standard excl_radius + std threshold) ──
        mask_center = _compute_excl_mask(c_np_full, excl_center, excl_radius, excl_mode)
        if std_excl_min > 0:
            mask_std_thr = ts_scalar > std_excl_min
            mask_std = mask_center & mask_std_thr
        else:
            mask_std = mask_center
        c_excl_std = c_np_full.copy()
        c_excl_std[:, mask_std] = float("nan")
        outputs_excl_std = dict(outputs_raw)
        outputs_excl_std["c"] = torch.from_numpy(c_excl_std).unsqueeze(0).to(device).to(outputs_raw["c"].dtype)
        angle_excl, dist_excl, _ = run_pnp(outputs_excl_std, gtbbox, qgt, rgt)

        # ── CORRECTED ──
        if corr_excl:
            raw_for_alpha = outputs_excl_std
        else:
            raw_for_alpha = outputs_raw
        coords_corr, _ = correct_coords(
            raw_for_alpha["c"].clone(), raw_for_alpha["logl"].clone(),
            raw_for_alpha["loga"].clone(), raw_for_alpha["logb"].clone(),
            alpha_table, std_min, std_max)
        outputs_corr_std = dict(outputs_raw)
        outputs_corr_std["c"] = coords_corr
        angle_corr, dist_corr, _ = run_pnp(outputs_corr_std, gtbbox, qgt, rgt)

        # ── Sweep: ratio only ──
        for radius in sweep_r:
            r3 = (radius, radius, radius)
            mask_c = _compute_excl_mask(c_np_full, excl_center, r3, excl_mode)
            ct_ratio = float(mask_c.sum()) / total_valid if total_valid > 0 else 0.0
            if std_excl_min > 0:
                mask = mask_c & (ts_scalar > std_excl_min)
            else:
                mask = mask_c
            ct_std_ratio = float(mask.sum()) / total_valid if total_valid > 0 else 0.0
            rows.append({
                "radius": radius,
                "ct_ratio": ct_ratio,
                "ct_std_ratio": ct_std_ratio,
                "excl_ratio": ct_std_ratio,
                "angle_base": angle_base, "angle_excl": angle_excl, "angle_corr": angle_corr,
                "dist_base": dist_base, "dist_excl": dist_excl, "dist_corr": dist_corr,
            })
        n_images += 1

    # save CSV
    csv_path = os.path.join(out_dir, f"{split}_excl_ratio_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  → {csv_path} ({len(rows)} points)")

    # plot
    _plot_sweep(rows, out_dir, split)


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

def _plot_sweep(rows, out_dir, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radii_uniq = sorted(set(r["radius"] for r in rows))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(radii_uniq)))

    shape_info = [
        ("base", "o", "BASELINE"),
        ("excl", "^", "EXCLUDED"),
        ("corr", "s", "CORRECTED"),
    ]

    for ykey, ylabel, fname in [
        ("angle", "Angle Error (deg)", "excl_ratio_vs_angle"),
        ("dist", "Distance Error", "excl_ratio_vs_dist"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for ki, rad in enumerate(radii_uniq):
            pts = [r for r in rows if r["radius"] == rad]
            for mode, marker, label in shape_info:
                rr = np.array([r["excl_ratio"] for r in pts])
                vv = np.array([r[f"{ykey}_{mode}"] for r in pts])
                vv = vv[np.isfinite(vv)]
                if len(vv) < 2:
                    continue
                ax.scatter(rr, vv, s=10, color=colors[ki], marker=marker, alpha=0.4)

        from matplotlib.lines import Line2D
        shape_leg = [Line2D([0], [0], marker=m, color="gray", linestyle="none", markersize=8, label=l)
                     for _, m, l in shape_info]
        leg1 = ax.legend(handles=shape_leg, fontsize=7, loc="lower left")
        ax.add_artist(leg1)

        pick_r = [0, len(radii_uniq)//4, len(radii_uniq)//2, 3*len(radii_uniq)//4, len(radii_uniq)-1]
        color_leg = [Line2D([0], [0], marker="o", color=colors[i], linestyle="none", markersize=8,
                            label=f"r={radii_uniq[i]:.2f}") for i in pick_r if i < len(radii_uniq)]
        leg2 = ax.legend(handles=color_leg, fontsize=7, loc="lower right")
        ax.add_artist(leg2)

        ax.set_xlabel("Exclusion ratio")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{split}: {ylabel} vs exclusion ratio")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{split}_{fname}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"  -> {save_path}")


def _plot_pred_vs_std(std_preds, std_ts, std_epi, std_alea, out_dir, split):
    """9-panel scatter: rows=(total, epi, alea) x cols=(x, y, z), x=pred, y=std."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.concatenate(std_preds)  # flat array: x then y then z
    ts   = np.concatenate(std_ts)
    epi  = np.concatenate(std_epi)
    alea = np.concatenate(std_alea)
    n_total = len(pred)
    n_per_ax = n_total // 3

    axis_names = ["x", "y", "z"]
    std_sets = [("total", ts), ("epistemic", epi), ("aleatoric", alea)]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex="col", sharey="row")
    for ri, (std_name, std_arr) in enumerate(std_sets):
        for ci, ax_name in enumerate(axis_names):
            ax = axes[ri, ci]
            ax_idx = ci
            px = pred[ax_idx::3]
            sy = std_arr[ax_idx::3]
            # subsample to 20k for performance
            if len(px) > 20000:
                ii = np.random.choice(len(px), 20000, replace=False)
                px = px[ii]; sy = sy[ii]
            ax.scatter(px, sy, s=1, alpha=0.3, color="#1f77b4", rasterized=True)
            if ri == 2:
                ax.set_xlabel(f"pred_{ax_name}")
            if ci == 0:
                ax.set_ylabel(f"{std_name} std")
            ax.grid(True, alpha=0.15)
    fig.suptitle(f"{split}: prediction vs uncertainty (rows=total/epistemic/aleatoric, cols=x/y/z)", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{split}_pred_vs_std.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  -> {save_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha-corrected PnP evaluation on sunlamp/lightbox")
    parser.add_argument("--uuid", default=None,
                        help="Model UUID (default: pairwise.uuid_1 from config.yaml)")
    parser.add_argument("--alpha_csv", default=os.path.join(PROJECT_ROOT, "outputs", "alpha_cali", "alpha_cali.csv"))
    parser.add_argument("--splits", nargs="*", default=["sunlamp", "lightbox"])
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--std_min", type=float, default=0.0,
                        help="Per-axis total_std lower bound (pixels outside → NaN)")
    parser.add_argument("--std_max", type=float, default=10.0,
                        help="Per-axis total_std upper bound (pixels outside → NaN)")
    parser.add_argument("--excl_cx", type=float, default=0.045, help="Exclusion zone center X")
    parser.add_argument("--excl_cy", type=float, default=0.057, help="Exclusion zone center Y")
    parser.add_argument("--excl_cz", type=float, default=0.16, help="Exclusion zone center Z")
    parser.add_argument("--excl_rx", type=float, default=0.1, help="Exclusion zone radius X")
    parser.add_argument("--excl_ry", type=float, default=0.1, help="Exclusion zone radius Y")
    parser.add_argument("--excl_rz", type=float, default=0.1, help="Exclusion zone radius Z")
    parser.add_argument("--corr_excl", action="store_true", default=False,
                        help="Apply exclusion filter to CORRECTED mode as well")
    parser.add_argument("--excl_mode", choices=["and", "or"], default="or",
                        help="Exclusion mode: all axes (and) or any axis (or)")
    parser.add_argument("--excl_sweep_r", nargs="*", type=float, default=None,
                        help="Exclusion radius sweep (enables ratio vs error analysis)")
    parser.add_argument("--no_sweep", action="store_true", default=False,
                        help="Disable radius sweep, use standard single-excl mode")
    parser.add_argument("--std_excl_min", type=float, default=0.01,
                        help="Only exclude pixels with total_std > this value")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    excl_center = (args.excl_cx, args.excl_cy, args.excl_cz) if args.excl_cx is not None else None
    excl_radius = (args.excl_rx, args.excl_ry, args.excl_rz) if args.excl_cx is not None else None

    # DEFAULT_SWEEP = [0.02, 0.05, 0.10, 0.15,0.2]
    DEFAULT_SWEEP = [0.10, 0.15,0.2, 0.25,0.3]
    if args.no_sweep:
        sweep_r = None
    elif args.excl_sweep_r:
        sweep_r = args.excl_sweep_r
    else:
        sweep_r = DEFAULT_SWEEP

    evaluate(args.alpha_csv, args.splits, args.max_samples, args.std_min, args.std_max,
             args.device, args.uuid, excl_center, excl_radius, args.corr_excl, args.excl_mode, sweep_r, args.std_excl_min)
