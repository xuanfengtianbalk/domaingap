"""alpha_calibration.py — Alpha calibration on validation via FGSM attack.

Stores 2D alpha lookup table (same structure as alpha_joint_profile) plus
per-ε robustness metrics (angle/distance error, total_std change).

Run: python analysis/alpha_calibration.py
"""

from __future__ import annotations
import sys, os, json, csv, yaml, math, argparse, numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dinov3_main"))

from analysis_utils import load_model, build_dataloader, extract_full
from utils_datasets.speedplus_utils_main.utils import Camera
from post_process import pose_calculats_from_coors, compute_pose_error, to_pnp_coors
from math_.q_ import quatProduct  # noqa: F401


# ── epsilon sampling ─────────────────────────────────────────────────────────

def sample_epsilon():
    """Return default epsilon values for FGSM attack."""
    return [0.001, 0.005, 0.01, 0.05, 0.1]


# ── FGSM attack ──────────────────────────────────────────────────────────────

def fgsm_attack(model, image: torch.Tensor, coors_gt: torch.Tensor,
                mask_gt: torch.Tensor, epsilon: float) -> torch.Tensor:
    """FGSM: image + ε · sign(∇ MSE(pred[mask], GT[mask])).

    Args:
        model: DER model
        image: clean input (1, 3, H, W), clamped to [0, 1]
        coors_gt: ground-truth coordinates (1, 3, H, W) or (3, H, W)
        mask_gt: binary mask (H, W) or (1, H, W), True where valid
        epsilon: perturbation magnitude

    Returns:
        adversarial image (1, 3, H, W), clamped to [0, 1]
    """
    device = image.device
    X = image.clone().detach().requires_grad_(True)
    gt_dev = coors_gt.to(device)

    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        outputs = model(X)

    pred = outputs["c"].float()                                     # (1, 3, H, W)

    if mask_gt.dim() == 2:
        mask = mask_gt.to(device).bool()
    else:
        mask = mask_gt.squeeze(0).to(device).bool()

    loss = torch.tensor(0.0, device=device)
    for b in range(X.shape[0]):
        mb = mask if mask.dim() == 2 else mask[b]
        if mb.sum() == 0:
            continue
        loss = loss + F.mse_loss(
            pred[b, :, mb].float(),
            gt_dev[b, :, mb].float() if gt_dev.dim() == 4 else gt_dev[:, mb].float(),
        )

    X.grad = None
    loss.backward()

    adv = X + epsilon * X.grad.sign()
    adv = torch.clamp(adv, 0.0, 1.0)
    return adv.detach()


# ── pose metrics (PnP + angle/dist) ──────────────────────────────────────────

def compute_pose_one(outputs_raw: dict, gtbbox: torch.Tensor, K: np.ndarray,
                     qgt: torch.Tensor, rgt: torch.Tensor) -> tuple:
    """Run PnP on raw model outputs, return (angle_deg, dist_abs, n_valid_pix, is_true)."""
    activate = torch.nn.Sigmoid()
    coormap = outputs_raw["c"].clone().detach()
    mask_bool = (activate(outputs_raw["mask"]) > 0.5).expand_as(coormap).cpu()
    coormap[~mask_bool] = float("nan")
    coormap_np = to_pnp_coors(coormap.squeeze())
    n_valid = (~np.isnan(coormap_np[:, :, 0])).sum()
    gtb = gtbbox.cpu().detach().numpy()
    try:
        is_true, qvecs, tvecs = pose_calculats_from_coors(K, coormap_np, gtb)
    except Exception:
        is_true, qvecs, tvecs = False, None, None
    qg = qgt.squeeze().cpu()
    rg = rgt.squeeze().cpu()
    err_ori_deg, _, err_r_abs, _, _, _ = compute_pose_error(qvecs, tvecs, qg, rg, is_true)
    return float(err_ori_deg), float(err_r_abs), int(n_valid), is_true


# ── total_std from extract_full result ───────────────────────────────────────

def _extract_from_raw(outputs_raw: dict, model_type: str) -> dict:
    """Extract coords + total_std from raw model output (no model re-run)."""
    result = {}
    if model_type == "coordinates_DER":
        result["coords"] = outputs_raw["c"].cpu()
        logl = outputs_raw["logl"].cpu().numpy()
        loga = outputs_raw["loga"].cpu().numpy()
        logb = outputs_raw["logb"].cpu().numpy()
        a = np.exp(loga) + 1.0 + 1e-6
        b = np.exp(logb) + 1e-6
        v = np.exp(logl) + 1e-6
        result["alea_var"] = b / (a - 1 + 1e-12)
        result["epi_var"] = b / ((a - 1 + 1e-12) * (v + 1e-12))
    return result


def _total_std_from_full(full: dict) -> np.ndarray:
    epi = full.get("epi_var")
    alea = full.get("alea_var")
    if epi is not None and alea is not None:
        return np.sqrt(np.maximum(epi + alea, 0.0))
    elif epi is not None:
        return np.sqrt(np.maximum(epi, 0.0))
    return None


# ── main calibration ─────────────────────────────────────────────────────────

def calibrate(uuid: str, epsilons: list, n_ts_bins: int = 10,
              max_samples: int = 1000, device: str = "cuda:0"):
    """Run calibration and save results."""

    import yaml as _yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f)
    gr = cfg["gt_ranges"]
    step = gr["bin_step"]

    # Load model
    from analysis_utils import get_model_type_from_traininfo
    mt, bb = get_model_type_from_traininfo(uuid)
    model_type = mt[0]
    print(f"Model: {model_type} on {bb}")
    model, bc = load_model(uuid, model_type, bb, device)
    model.eval()

    # Dataloader
    dl = build_dataloader(uuid, "validation", max_samples=max_samples, batch_size=1)

    # Accumulators
    angle_list = {str(e): [] for e in epsilons}
    dist_list  = {str(e): [] for e in epsilons}
    ts_clean_list = {str(e): [] for e in epsilons}
    ts_adv_list   = {str(e): [] for e in epsilons}

    alpha_preds, alpha_ts, alpha_gts = [], [], []      # per-pixel for alpha table

    n_images = 0
    for samples, targets in tqdm(dl, desc="Calibrating", ncols=80):
        image = samples.to(device)                                  # (1, 3, H, W)
        mask_gt = targets["mask_gt"][0]                             # (H, W)
        coors_gt = targets["coors_gt"]                              # (1, 3, H, W)

        # --- clean forward ---
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs_raw_clean = model(image)
        full_clean = _extract_from_raw(outputs_raw_clean, model_type)
        total_std_clean = _total_std_from_full(full_clean)
        pred_clean = full_clean["coords"][0].numpy()                # (3, H, W)

        # --- collect alpha pixels (clean data) ---
        m_valid = (mask_gt > 0.5).numpy()
        if m_valid.sum() > 0 and total_std_clean is not None:
            ts_arr = total_std_clean.squeeze(0)                             # (3, H, W) if batched
            if ts_arr.ndim == 3:                                            # per-coord channel
                ts_pix = ts_arr[:, m_valid]
            else:                                                           # (H, W) scalar
                ts_pix = np.broadcast_to(ts_arr[m_valid][np.newaxis, :], (3, m_valid.sum()))
            for ax in range(3):
                p = pred_clean[ax, m_valid]
                g = coors_gt[0, ax, m_valid].numpy()
                alpha_preds.append(p)
                alpha_ts.append(ts_pix[ax])
                alpha_gts.append(g)

        # --- FGSM loop ---
        for eps in epsilons:
            adv = fgsm_attack(model, image, coors_gt, mask_gt, eps)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs_raw_adv = model(adv)
            full_adv = _extract_from_raw(outputs_raw_adv, model_type)
            total_std_adv = _total_std_from_full(full_adv)

            angle, dist, _, _ = compute_pose_one(outputs_raw_adv,
                                           torch.round(targets["boxes"].squeeze(0)),
                                           Camera.K,
                                           targets["q_gt"].squeeze(),
                                           targets["r_gt"].squeeze())

            k = str(eps)
            if not math.isnan(angle):
                angle_list[k].append(angle)
                dist_list[k].append(dist)
            if total_std_clean is not None:
                ts_clean_list[k].append(float(np.nanmean(total_std_clean)))
            if total_std_adv is not None:
                ts_adv_list[k].append(float(np.nanmean(total_std_adv)))

        n_images += 1

    print(f"\nProcessed {n_images} images, {len(alpha_preds)} alpha pixels")

    # --- robustness stats ---
    robustness = {}
    for eps in epsilons:
        k = str(eps)
        rob = {"n_valid": len(angle_list[k])}
        if rob["n_valid"] > 0:
            rob["angle_mean"] = float(np.mean(angle_list[k]))
            rob["angle_std"]  = float(np.std(angle_list[k]))
            rob["dist_mean"]  = float(np.mean(dist_list[k]))
            rob["dist_std"]   = float(np.std(dist_list[k]))
        if ts_clean_list[k]:
            rob["ts_clean_mean"] = float(np.mean(ts_clean_list[k]))
        if ts_adv_list[k]:
            rob["ts_adv_mean"] = float(np.mean(ts_adv_list[k]))
        robustness[k] = rob

    # --- build 2D alpha table per axis ---
    alpha_preds = np.concatenate(alpha_preds).ravel()
    alpha_ts    = np.concatenate(alpha_ts).ravel()
    alpha_gts   = np.concatenate(alpha_gts).ravel()
    n_total = len(alpha_preds)
    n_per_ax = n_total // 3
    print(f"  table pixels: {n_total} ({n_per_ax} per axis)")

    alpha_table = {}
    for ax_idx, key in enumerate(["x", "y", "z"]):
        pred_vals = alpha_preds[ax_idx::3] if len(alpha_preds) % 3 == 0 else alpha_preds
        ts_vals   = alpha_ts[ax_idx::3] if len(alpha_ts) % 3 == 0 else alpha_ts
        gt_vals   = alpha_gts[ax_idx::3] if len(alpha_gts) % 3 == 0 else alpha_gts

        eps_ = 1e-3
        valid = np.abs(pred_vals) >= eps_
        if valid.sum() < 10:
            alpha_table[key] = {"grid": [], "total_std_edges": [], "pred_edges": []}
            continue

        pv, gv, tv = pred_vals[valid], gt_vals[valid], ts_vals[valid]
        alpha_v = (gv - pv) / (pv * tv + 1e-12)

        ts_edges = np.percentile(tv, np.linspace(0, 100, n_ts_bins + 1))
        gr_ax = gr[key]
        inner = np.arange(gr_ax[0], gr_ax[1] + step * 0.5, step)
        p_edges = np.concatenate([[-np.inf], inner, [np.inf]])

        grid = []
        for i, (tlo, thi) in enumerate(zip(ts_edges[:-1], ts_edges[1:])):
            m_t = (tv >= tlo) & (tv < thi)
            for j, (plo, phi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
                m_p = (pv >= plo) & (pv < phi)
                m = m_t & m_p
                if m.sum() < 5:
                    continue
                a = alpha_v[m]
                grid.append({
                    "total_std_bin": i,  "total_std_lo": float(tlo),  "total_std_hi": float(thi),
                    "pred_bin":      j,  "pred_lo":      float(plo),  "pred_hi":      float(phi),
                    "n": int(m.sum()),
                    "mean_alpha": float(a.mean()), "std_alpha": float(a.std()),
                    "mean_GT": float(gv[m].mean()),
                    "mean_pred": float(pv[m].mean()),
                    "mean_total_std": float(tv[m].mean()),
                    "axis": key,
                })

        alpha_table[key] = {
            "grid": grid,
            "total_std_edges": [float(e) for e in ts_edges],
            "pred_edges": [float(e) for e in p_edges],
        }

    # --- write outputs ---
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "alpha_cali")
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "metadata": {
            "uuid":             uuid,
            "epsilons":         epsilons,
            "n_images":         n_images,
            "n_alpha_pixels":   int(len(alpha_preds)),
            "n_total_std_bins": n_ts_bins,
        },
        "robustness": robustness,
        "alpha_table": alpha_table,
    }

    json_path = os.path.join(out_dir, "alpha_cali.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {json_path}")

    # CSV
    csv_rows = []
    for key in ["x", "y", "z"]:
        for cell in alpha_table[key]["grid"]:
            csv_rows.append(cell)
    if csv_rows:
        csv_path = os.path.join(out_dir, "alpha_cali.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"  → {csv_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha calibration via FGSM on validation")
    parser.add_argument("--uuid", default="e24d72fb-b4d7-4c40-b00a-a4aa2f8213e8",
                        help="DER model UUID")
    parser.add_argument("--epsilons", nargs="*", type=float,
                        default=None, help="FGSM epsilon values")
    parser.add_argument("--n_ts_bins", type=int, default=10,
                        help="Number of total_std percentile bins")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max validation images")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    epsilons = args.epsilons if args.epsilons else sample_epsilon()
    print(f"Epsilons: {epsilons}")
    calibrate(args.uuid, epsilons, args.n_ts_bins, args.max_samples, args.device)
