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
from scipy.stats import norm, beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dinov3_main"))

from analysis_utils import load_model, build_dataloader, extract_full
from utils_datasets.speedplus_utils_main.utils import Camera
from post_process import pose_calculats_from_coors, compute_pose_error, to_pnp_coors
from math_.q_ import quatProduct  # noqa: F401


# ── epsilon sampling ─────────────────────────────────────────────────────────

def sample_epsilon(eps):
    """Return default epsilon values for FGSM attack."""
    if eps is None:
        return [1.8, 1.9, 2.0, 2.1, 2.2]
    return eps


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
    # adv = torch.clamp(adv, 0.0, 1.0)
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
        is_true, qvecs, tvecs = pose_calculats_from_coors(K, coormap_np, gtb[0])
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


def compute_calibration_torch(pre, gt, pre_delta, conf=10, alpha_ci=0.05, return_errors=True):
    """Reliability calibration: alpha vs actual coverage.

    pre:        Tensor [N], model predictions
    gt:         Tensor [N], ground truth
    pre_delta:  Tensor [N], model total_std
    conf:       int, number of confidence levels (default 10 → 0.1 ~ 1.0)
    alpha_ci:   float, Clopper-Pearson CI alpha (default 0.05 → 95%)
    return_errors: bool, return MAE / RMSCE / MCE

    Returns: alphas, coverages, ci_lower, ci_upper, [calib_errors]
    """
    alphas = torch.linspace(0, 1.0, conf + 1)
    coverages = []
    ci_lower = []
    ci_upper = []

    N = pre.shape[0]
    pre = pre.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    pre_delta = pre_delta.detach().cpu().numpy()

    for alpha in alphas:
        z = norm.ppf(0.5 + float(alpha) / 2)
        lower = pre - z * pre_delta
        upper = pre + z * pre_delta
        covers = (gt >= lower) & (gt <= upper)
        count = int(covers.sum())
        coverage = count / N
        coverages.append(coverage)
        ci_l = beta.ppf(alpha_ci / 2, count, N - count + 1) if count > 0 else 0.0
        ci_u = beta.ppf(1 - alpha_ci / 2, count + 1, N - count) if count < N else 1.0
        ci_lower.append(ci_l)
        ci_upper.append(ci_u)

    alphas = torch.tensor([float(a) for a in alphas])
    coverages = torch.tensor(coverages)
    ci_lower = torch.tensor(ci_lower)
    ci_upper = torch.tensor(ci_upper)

    calib_errors = {}
    if return_errors:
        diff = coverages - alphas
        calib_errors["MAE"] = torch.mean(torch.abs(diff)).item()
        calib_errors["RMSCE"] = torch.sqrt(torch.mean(diff ** 2)).item()
        calib_errors["MCE"] = torch.max(torch.abs(diff)).item()

    if return_errors:
        return alphas, coverages, ci_lower, ci_upper, calib_errors
    else:
        return alphas, coverages, ci_lower, ci_upper


def _plot_calibration_curve(calibration: dict, save_path: str):
    """Reliability diagram: alpha vs coverage, one curve per epsilon."""
    fig, ax = plt.subplots(figsize=(7, 5))
    eps_keys = sorted(calibration.keys(), key=float)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(eps_keys)))
    for ki, eps_k in enumerate(eps_keys):
        c = calibration[eps_k]
        if c.get("n_pixels", 0) == 0:
            continue
        alphas = np.array(c["alphas"])
        coverages = np.array(c["coverages"])
        ci_lo = np.array(c["ci_lower"])
        ci_hi = np.array(c["ci_upper"])
        errs = c["errors"]
        label = f"ε={eps_k} (MAE={errs['MAE']:.3f})"
        ax.fill_between(alphas, ci_lo, ci_hi, color=colors[ki], alpha=0.08)
        ax.plot(alphas, coverages, "o-", color=colors[ki], markersize=4, linewidth=1.2, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect")
    ax.set_xlabel("Confidence level (alpha)")
    ax.set_ylabel("Actual coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_title("Reliability Calibration Curve (per FGSM epsilon)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _build_excl_pred_edges(gt_range, cx, rx, n_bins=10):
    """Build pred edges with exclusion zone removed, n_bins split across remaining ranges."""
    full_lo, full_hi = gt_range[0], gt_range[1]
    excl_lo, excl_hi = cx - rx, cx + rx

    ranges = []
    if excl_lo > full_lo:
        ranges.append((full_lo, excl_lo))
    if excl_hi < full_hi:
        ranges.append((excl_hi, full_hi))

    if not ranges:
        return np.array([-np.inf, np.inf])

    total_w = sum(hi - lo for lo, hi in ranges)
    edges = [-np.inf]
    remaining = n_bins
    for i, (lo, hi) in enumerate(ranges):
        if i == len(ranges) - 1:
            n = remaining
        else:
            n = max(1, int(round((hi - lo) / total_w * n_bins)))
            remaining -= n
        seg_edges = np.linspace(lo, hi, n + 1)
        edges.extend(seg_edges.tolist())
    edges.append(np.inf)
    return np.array(edges)
    """Reliability diagram: alpha vs coverage, one curve per epsilon."""
    fig, ax = plt.subplots(figsize=(7, 5))
    eps_keys = sorted(calibration.keys(), key=float)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(eps_keys)))
    for ki, eps_k in enumerate(eps_keys):
        c = calibration[eps_k]
        if c.get("n_pixels", 0) == 0:
            continue
        alphas = np.array(c["alphas"])
        coverages = np.array(c["coverages"])
        ci_lo = np.array(c["ci_lower"])
        ci_hi = np.array(c["ci_upper"])
        errs = c["errors"]
        label = f"ε={eps_k} (MAE={errs['MAE']:.3f})"
        ax.fill_between(alphas, ci_lo, ci_hi, color=colors[ki], alpha=0.08)
        ax.plot(alphas, coverages, "o-", color=colors[ki], markersize=4, linewidth=1.2, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect")
    ax.set_xlabel("Confidence level (alpha)")
    ax.set_ylabel("Actual coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_title("Reliability Calibration Curve (per FGSM epsilon)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ── MLP independent training ───────────────────────────────────────────────────

def _train_mlp_independent(uuid, model, device, aug_type, mlp_epochs, mlp_batch, mlp_lr,
                           excl_suffix, mode, out_dir, max_samples,
                           mlp_freq_enc=True, mlp_excl=True, mlp_loss="l2"):
    """Train MLP on augmix data using independent dataloader."""
    from alpha_mlp import UnifiedCorrectionMLP
    from utils_datasets.speedplus_utils_main.space_aug import SpaceAugTransform
    from analysis_utils import build_dataloader

    if mlp_loss == "l1":
        loss_fn = torch.nn.functional.l1_loss
    elif mlp_loss == "smooth_l1":
        loss_fn = torch.nn.functional.smooth_l1_loss
    else:
        loss_fn = torch.nn.functional.mse_loss

    augmentor = SpaceAugTransform(aug_type, styleaug_p=0.0)
    mlp_model = UnifiedCorrectionMLP(use_freq_enc=mlp_freq_enc).to(device)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=mlp_lr)

    dl = build_dataloader(uuid, "validation", max_samples=max_samples, batch_size=1)
    imgs = list(dl)
    n_total = len(imgs)
    n_train = int(0.8 * n_total)

    n_steps_per_epoch = max(1, n_train // mlp_batch)
    freq_str = "raw" if not mlp_freq_enc else "freq"
    excl_str = "excl" if mlp_excl else "noexcl"
    print(f"  MLP [{freq_str},{excl_str},{mlp_loss}]: {n_total} images ({n_train} train), {mlp_epochs} epochs, {n_steps_per_epoch} steps/epoch")

    eps = 1e-3
    cx, cy, cz = 0.045, 0.057, 0.16
    rx, ry, rz = 0.05, 0.05, 0.05

    for epoch in range(mlp_epochs):
        mlp_model.train()
        total_loss, total_raw, n_steps = 0.0, 0.0, 0
        perm = torch.randperm(n_train)

        for start in range(0, n_train, mlp_batch):
            end = min(start + mlp_batch, n_train)
            idx = perm[start:end].tolist()

            all_px, all_py, all_pz = [], [], []
            all_tx, all_ty, all_tz = [], [], []
            all_gx, all_gy, all_gz = [], [], []

            for i in idx:
                samples, targets = imgs[i]
                image = samples.to(device)
                mask = targets["mask_gt"][0] > 0.5

                # augmix
                img_np = (image.cpu().squeeze(0).permute(1,2,0).numpy() * 255).clip(0, 255).astype(np.uint8)
                aug_np = augmentor(image=img_np)["image"]
                aug_tensor = torch.from_numpy(aug_np.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)

                with torch.no_grad(), torch.amp.autocast("cuda"):
                    out = model(aug_tensor)

                pred = out["c"].squeeze(0).cpu().numpy()
                gt   = targets["coors_gt"][0].numpy()

                logl = out["logl"].squeeze(0).cpu().numpy()
                loga = out["loga"].squeeze(0).cpu().numpy()
                logb = out["logb"].squeeze(0).cpu().numpy()
                a = np.exp(loga) + 1.0 + 1e-6
                b_v = np.exp(logb) + 1e-6
                v = np.exp(logl) + 1e-6
                epi_var = b_v / ((a - 1 + 1e-12) * (v + 1e-12))
                alea_var = b_v / (a - 1 + 1e-12)
                ts_3d = np.sqrt(np.maximum(epi_var + alea_var, 0.0))

                m = mask.numpy()
                vv = (np.abs(pred[0,m]) >= eps) & (np.abs(pred[1,m]) >= eps) & (np.abs(pred[2,m]) >= eps)
                if not vv.any():
                    continue
                # exclusion filter (optional)
                if mlp_excl:
                    vv &= (np.abs(pred[0,m] - cx) >= rx) | (np.abs(pred[1,m] - cy) >= ry) | (np.abs(pred[2,m] - cz) >= rz)
                if not vv.any():
                    continue

                all_px.append(pred[0,m][vv]); all_py.append(pred[1,m][vv]); all_pz.append(pred[2,m][vv])
                all_tx.append(ts_3d[0,m][vv]); all_ty.append(ts_3d[1,m][vv]); all_tz.append(ts_3d[2,m][vv])
                all_gx.append(gt[0,m][vv]); all_gy.append(gt[1,m][vv]); all_gz.append(gt[2,m][vv])

            if not all_px:
                continue

            px_t = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in all_px]).to(device)
            py_t = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in all_py]).to(device)
            pz_t = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in all_pz]).to(device)
            tx_t = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in all_tx]).to(device)
            ty_t = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in all_ty]).to(device)
            tz_t = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in all_tz]).to(device)
            gt_b = torch.stack([
                torch.cat([torch.tensor(x, dtype=torch.float32) for x in all_gx]),
                torch.cat([torch.tensor(x, dtype=torch.float32) for x in all_gy]),
                torch.cat([torch.tensor(x, dtype=torch.float32) for x in all_gz]),
            ], dim=-1).to(device)

            raw_pred_train = torch.stack([px_t.squeeze(-1), py_t.squeeze(-1), pz_t.squeeze(-1)], dim=-1)
            pred_out = mlp_model(tx_t, px_t, ty_t, py_t, tz_t, pz_t)
            loss = loss_fn(pred_out, gt_b)
            raw_loss_train = loss_fn(raw_pred_train, gt_b).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_raw += raw_loss_train
            n_steps += 1

        # validation
        if n_steps > 0:
            mlp_model.eval()
            val_px, val_py, val_pz, val_tx, val_ty, val_tz, val_gx, val_gy, val_gz = [], [], [], [], [], [], [], [], []
            for i in range(n_train, n_total):
                samples, targets = imgs[i]
                image = samples.to(device)
                mask = targets["mask_gt"][0] > 0.5
                img_np = (image.cpu().squeeze(0).permute(1,2,0).numpy() * 255).clip(0, 255).astype(np.uint8)
                aug_np = augmentor(image=img_np)["image"]
                aug_tensor = torch.from_numpy(aug_np.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    out = model(aug_tensor)
                pred = out["c"].squeeze(0).cpu().numpy()
                gt   = targets["coors_gt"][0].numpy()
                logl = out["logl"].squeeze(0).cpu().numpy()
                loga = out["loga"].squeeze(0).cpu().numpy()
                logb = out["logb"].squeeze(0).cpu().numpy()
                a_v = np.exp(loga) + 1.0 + 1e-6
                bv = np.exp(logb) + 1e-6
                vl = np.exp(logl) + 1e-6
                ep = bv / ((a_v - 1 + 1e-12) * (vl + 1e-12))
                al = bv / (a_v - 1 + 1e-12)
                ts = np.sqrt(np.maximum(ep + al, 0.0))
                m = mask.numpy()
                vv = (np.abs(pred[0,m]) >= eps) & (np.abs(pred[1,m]) >= eps) & (np.abs(pred[2,m]) >= eps)
                if not vv.any(): continue
                if mlp_excl:
                    vv &= (np.abs(pred[0,m] - cx) >= rx) | (np.abs(pred[1,m] - cy) >= ry) | (np.abs(pred[2,m] - cz) >= rz)
                if not vv.any(): continue
                val_px.append(pred[0,m][vv]); val_py.append(pred[1,m][vv]); val_pz.append(pred[2,m][vv])
                val_tx.append(ts[0,m][vv]); val_ty.append(ts[1,m][vv]); val_tz.append(ts[2,m][vv])
                val_gx.append(gt[0,m][vv]); val_gy.append(gt[1,m][vv]); val_gz.append(gt[2,m][vv])

            if val_px:
                vtx = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in val_tx]).to(device)
                vpx = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in val_px]).to(device)
                vty = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in val_ty]).to(device)
                vpy = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in val_py]).to(device)
                vtz = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in val_tz]).to(device)
                vpz = torch.cat([torch.tensor(x, dtype=torch.float32).reshape(-1,1) for x in val_pz]).to(device)
                vgt = torch.stack([
                    torch.cat([torch.tensor(x, dtype=torch.float32) for x in val_gx]),
                    torch.cat([torch.tensor(x, dtype=torch.float32) for x in val_gy]),
                    torch.cat([torch.tensor(x, dtype=torch.float32) for x in val_gz]),
                ], dim=-1).to(device)
                with torch.no_grad():
                    val_out = mlp_model(vtx, vpx, vty, vpy, vtz, vpz)
                    val_loss = loss_fn(val_out, vgt).item()
                    raw_pred = torch.stack([vpx.squeeze(-1), vpy.squeeze(-1), vpz.squeeze(-1)], dim=-1)
                    raw_loss = loss_fn(raw_pred, vgt).item()
            else:
                val_loss = float("nan")
                raw_loss = float("nan")

            if epoch % 10 == 0 or epoch == mlp_epochs - 1:
                print(f"    epoch {epoch:3d}: train={total_loss/n_steps:.4f}(raw={total_raw/n_steps:.4f}) val={val_loss:.4f}(raw={raw_loss:.4f})")

    # save
    model_name = f"alpha_mlp_{mode}_{aug_type}{excl_suffix}"
    pt_path = os.path.join(out_dir, f"{model_name}.pt")
    torch.save(mlp_model.state_dict(), pt_path)
    print(f"  → {pt_path}")


# ── main calibration ─────────────────────────────────────────────────────────

def calibrate(uuid: str, epsilons: list, n_ts_bins: int = 10,
              max_samples: int = 1000, std_bins: list = None, device: str = "cuda:0",
              excl_cx: float = None, excl_cy: float = None, excl_cz: float = None,
              excl_rx: float = None, excl_ry: float = None, excl_rz: float = None,
              mode: str = "fgsm", aug_type: str = None,
              train_mlp: bool = False, mlp_epochs: int = 100, mlp_lr: float = 1e-3,
              mlp_batch: int = 16, mlp_freq_enc: bool = True, mlp_excl: bool = True,
              mlp_loss: str = "l2"):
    """Run calibration and save results."""

    import yaml as _yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f)
    gr = cfg["gt_ranges"]
    step = gr["bin_step"]
    trim_pct = gr.get("trim_pct", 0.1)
    n_pred_bins = gr.get("n_pred_bins", 15)
    excl_n_pred_bins = gr.get("excl_n_pred_bins", 10)
    if std_bins is None:
        std_bins = np.linspace(0.01, 5, 20).tolist()

    # Load model
    from analysis_utils import get_model_type_from_traininfo
    mt, bb = get_model_type_from_traininfo(uuid)
    model_type = mt[0]
    print(f"Model: {model_type} on {bb}")
    model, bc = load_model(uuid, model_type, bb, device)
    model.eval()

    # Dataloader
    dl = build_dataloader(uuid, "validation", max_samples=max_samples, batch_size=1)

    # --- early return: MLP-only mode ---
    if train_mlp:
        # ensure aug_type is set
        _at = aug_type
        if _at is None:
            _train_cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "cfg.yaml")
            with open(_train_cfg_path) as _f:
                _train_cfg = _yaml.safe_load(_f)
            _at = _train_cfg.get("AUG_TYPE", "augmix")
        _excl_active = all(x is not None for x in [excl_cx, excl_cy, excl_cz, excl_rx, excl_ry, excl_rz])
        _excl_suffix = ""
        if _excl_active:
            _excl_suffix = f"_excl_cx{excl_cx}_cy{excl_cy}_cz{excl_cz}_rx{excl_rx}_ry{excl_ry}_rz{excl_rz}"
        _out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "alpha_cali")
        os.makedirs(_out_dir, exist_ok=True)
        _train_mlp_independent(uuid, model, device, _at, mlp_epochs, mlp_batch, mlp_lr,
                               _excl_suffix, mode, _out_dir, max_samples, mlp_freq_enc, mlp_excl, mlp_loss)
        return


    # --- augmix transform (if mode == "augmix") ---
    if aug_type is None:
        train_cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "cfg.yaml")
        with open(train_cfg_path) as f:
            train_cfg = _yaml.safe_load(f)
        aug_type = train_cfg.get("AUG_TYPE", "augmix")
    if mode == "augmix":
        from utils_datasets.speedplus_utils_main.space_aug import SpaceAugTransform
        aug_transform = SpaceAugTransform(aug_type, styleaug_p=0.5)
        epsilons = ["0"]
        print(f"Mode: augmix, aug_type={aug_type}")

    # Accumulators
    angle_list = {str(e): [] for e in epsilons}
    dist_list  = {str(e): [] for e in epsilons}
    ts_adv_list   = {str(e): [] for e in epsilons}

    # Per-epsilon calibration accumulators (for CSV alpha table only)
    max_calib_pix = 200_000
    calib_preds = {str(e): [[], [], []] for e in epsilons}
    calib_gts   = {str(e): [[], [], []] for e in epsilons}
    calib_ts    = {str(e): [[], [], []] for e in epsilons}
    calib_counts = {str(e): [0, 0, 0] for e in epsilons}

    n_images = 0
    for samples, targets in tqdm(dl, desc="Calibrating", ncols=80):
        image = samples.to(device)                                  # (1, 3, H, W)
        mask_gt = targets["mask_gt"][0]                             # (H, W)
        coors_gt = targets["coors_gt"]                              # (1, 3, H, W)
        m_valid = (mask_gt > 0.5).numpy()

        # --- attack loop ---
        if mode == "augmix":
            # Augmix: apply augmentation once, skip FGSM, no PnP
            img_np = (image.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            aug_img_np = aug_transform(image=img_np)["image"]
            aug_tensor = torch.from_numpy(aug_img_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs_raw_adv = model(aug_tensor)
            full_adv = _extract_from_raw(outputs_raw_adv, model_type)
            total_std_adv = _total_std_from_full(full_adv)

            if m_valid.sum() > 0 and total_std_adv is not None:
                pred_adv = full_adv["coords"][0].numpy()
                ts_arr_adv = total_std_adv.squeeze(0)
                if ts_arr_adv.ndim == 3:
                    ts_pix_adv = ts_arr_adv[:, m_valid]
                else:
                    ts_pix_adv = np.broadcast_to(ts_arr_adv[m_valid][np.newaxis, :], (3, m_valid.sum()))
                n_pix = int(m_valid.sum())
                k = "0"  # augmix has no epsilon
                for ax in range(3):
                    need = max_calib_pix - calib_counts[k][ax]
                    if need <= 0:
                        continue
                    take = min(n_pix, need)
                    if take < n_pix:
                        idx = np.random.choice(n_pix, take, replace=False)
                    else:
                        idx = slice(None)
                    calib_preds[k][ax].append(pred_adv[ax, m_valid][idx])
                    calib_gts[k][ax].append(coors_gt[0, ax, m_valid].numpy()[idx])
                    calib_ts[k][ax].append(ts_pix_adv[ax][idx])
                    calib_counts[k][ax] += take

            n_images += 1
            continue

        # --- FGSM loop ---
        for eps in epsilons:
            adv = fgsm_attack(model, image, coors_gt, mask_gt, eps)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs_raw_adv = model(adv)
            full_adv = _extract_from_raw(outputs_raw_adv, model_type)
            total_std_adv = _total_std_from_full(full_adv)

            # collect sampled calibration pixels (attacked)
            if m_valid.sum() > 0 and total_std_adv is not None:
                pred_adv = full_adv["coords"][0].numpy()
                ts_arr_adv = total_std_adv.squeeze(0)
                if ts_arr_adv.ndim == 3:
                    ts_pix_adv = ts_arr_adv[:, m_valid]
                else:
                    ts_pix_adv = np.broadcast_to(ts_arr_adv[m_valid][np.newaxis, :], (3, m_valid.sum()))
                n_pix = int(m_valid.sum())
                k = str(eps)
                for ax in range(3):
                    need = max_calib_pix - calib_counts[k][ax]
                    if need <= 0:
                        continue
                    take = min(n_pix, need)
                    if take < n_pix:
                        idx = np.random.choice(n_pix, take, replace=False)
                    else:
                        idx = slice(None)
                    calib_preds[k][ax].append(pred_adv[ax, m_valid][idx])
                    calib_gts[k][ax].append(coors_gt[0, ax, m_valid].numpy()[idx])
                    calib_ts[k][ax].append(ts_pix_adv[ax][idx])
                    calib_counts[k][ax] += take

            angle, dist, _, _ = compute_pose_one(outputs_raw_adv,
                                           torch.round(targets["boxes"].squeeze(0)),
                                           Camera.K,
                                           targets["q_gt"].squeeze(),
                                           targets["r_gt"].squeeze())

            k = str(eps)
            if not math.isnan(angle):
                angle_list[k].append(angle)
                dist_list[k].append(dist)
            if total_std_adv is not None:
                ts_adv_list[k].append(float(np.nanmean(total_std_adv)))

        n_images += 1

    # --- merge attacked data across epsilons for alpha table ---
    merged_p, merged_g, merged_t = [], [], []
    for ax in range(3):
        mp, mg, mt = [], [], []
        for eps in epsilons:
            k = str(eps)
            if calib_preds[k][ax]:
                mp.extend(calib_preds[k][ax])
                mg.extend(calib_gts[k][ax])
                mt.extend(calib_ts[k][ax])
        merged_p.append(mp)
        merged_g.append(mg)
        merged_t.append(mt)

    n_alpha = sum(len(x) for x in merged_p[0])
    print(f"\nProcessed {n_images} images, merged alpha pixels: {n_alpha} per axis")

    # --- per-axis center exclusion filter ---
    excl_active = all(x is not None for x in [excl_cx, excl_cy, excl_cz, excl_rx, excl_ry, excl_rz])
    excl_centers = [excl_cx, excl_cy, excl_cz]
    excl_radii  = [excl_rx, excl_ry, excl_rz]
    excl_suffix = ""
    eps_name = "_".join(str(e) for e in epsilons)
    if excl_active:
        excl_suffix = f"_excl_cx{excl_cx}_cy{excl_cy}_cz{excl_cz}_rx{excl_rx}_ry{excl_ry}_rz{excl_rz}"
        # flatten first, then filter
        mp_flat = [np.concatenate([np.atleast_1d(x) for x in merged_p[ax]]) for ax in range(3)]
        mg_flat = [np.concatenate([np.atleast_1d(x) for x in merged_g[ax]]) for ax in range(3)]
        mt_flat = [np.concatenate([np.atleast_1d(x) for x in merged_t[ax]]) for ax in range(3)]
        
        # per-axis keep for alpha table
        for ax in range(3):
            keep_ax = np.abs(mp_flat[ax] - excl_centers[ax]) >= excl_radii[ax]
            merged_p[ax] = [mp_flat[ax][keep_ax]]
            merged_g[ax] = [mg_flat[ax][keep_ax]]
            merged_t[ax] = [mt_flat[ax][keep_ax]]
        n_kept = len(merged_p[0][0])
        print(f"  exclusion filter: {n_kept} pixels/axis remain ({(1-n_kept/n_alpha)*100:.1f}% excluded)")

        # shared mask for supplementary analyses (pixel only if ALL axes survive)
        shared_keep = np.ones(len(mp_flat[0]), dtype=bool)
        for ax in range(3):
            shared_keep &= np.abs(mp_flat[ax] - excl_centers[ax]) >= excl_radii[ax]
        analysis_p_ax = [mp_flat[ax][shared_keep] for ax in range(3)]
        analysis_g_ax = [mg_flat[ax][shared_keep] for ax in range(3)]
        analysis_t_ax = [mt_flat[ax][shared_keep] for ax in range(3)]
        print(f"  shared mask for analyses: {shared_keep.sum()} pixels remain")

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
        if ts_adv_list[k]:
            rob["ts_adv_mean"] = float(np.mean(ts_adv_list[k]))
        robustness[k] = rob

    n_total = 0
    n_ts_bins_eff = 0
    alpha_table = {}
    if not train_mlp:
    # --- build 2D alpha table per axis (merged attacked data, fixed std edges) ---
        ts_edges = np.array([0.0] + list(std_bins) + [np.inf])
        n_ts_bins_eff = len(ts_edges) - 1

        alpha_table = {}
        for ax_idx, key in enumerate(["x", "y", "z"]):
            mp = np.concatenate([np.atleast_1d(x) for x in merged_p[ax_idx]])
            mg = np.concatenate([np.atleast_1d(x) for x in merged_g[ax_idx]])
            mt = np.concatenate([np.atleast_1d(x) for x in merged_t[ax_idx]])

            eps_ = 1e-3
            valid = np.abs(mp) >= eps_
            if valid.sum() < 10:
                alpha_table[key] = {"grid": [], "total_std_edges": [], "pred_edges": []}
                continue

            pv, gv, tv = mp[valid], mg[valid], mt[valid]
            alpha_v = (gv - pv) / (pv * tv + 1e-12)

            gr_ax = gr[key]
            if excl_active:
                p_edges = _build_excl_pred_edges(gr_ax, excl_centers[ax_idx], excl_radii[ax_idx], excl_n_pred_bins)
            else:
                inner = np.linspace(gr_ax[0], gr_ax[1], n_pred_bins + 1)
                p_edges = np.concatenate([[-np.inf], inner, [np.inf]])

            grid = []
            for i, (tlo, thi) in enumerate(zip(ts_edges[:-1], ts_edges[1:])):
                m_t = (tv >= tlo) & (tv < thi) if thi == np.inf else (tv >= tlo)
                for j, (plo, phi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
                    m_p = (pv >= plo) & (pv < phi)
                    m = m_t & m_p
                    if m.sum() < 5:
                        continue
                    a = alpha_v[m]
                    if m.sum() >= 10:
                        lo_a, hi_a = np.percentile(a, [trim_pct * 100, (1 - trim_pct) * 100])
                        a_ma = a[(a >= lo_a) & (a <= hi_a)]
                    else:
                        a_ma = a
                    grid.append({
                        "total_std_bin": i,  "total_std_lo": float(tlo),  "total_std_hi": float(thi),
                        "pred_bin":      j,  "pred_lo":      float(plo),  "pred_hi":      float(phi),
                        "n": int(m.sum()),
                        "mean_alpha": float(a_ma.mean()), "std_alpha": float(a.std()),
                        "mean_GT": float(gv[m].mean()),
                        "mean_pred": float(pv[m].mean()),
                        "mean_total_std": float(tv[m].mean()),
                        "axis": key,
                        "eps": eps_name,
                    })

            alpha_table[key] = {
                "grid": grid,
                "total_std_edges": [float(e) for e in ts_edges],
                "pred_edges": [float(e) for e in p_edges],
            }

        n_total = sum(len(mp) for mp in merged_p[0:1])
        print(f"  table: {sum(len(t['grid']) for t in alpha_table.values())} cells, {n_total} pixels/axis")

    # --- calibration reliability (alpha vs coverage per epsilon) ---
    calibration = {}
    for eps in epsilons:
        k = str(eps)
        cp_all, cg_all, ct_all = [], [], []
        for ax in range(3):
            if calib_preds[k][ax]:
                cp_all.append(np.concatenate(calib_preds[k][ax]))
                cg_all.append(np.concatenate(calib_gts[k][ax]))
                ct_all.append(np.concatenate(calib_ts[k][ax]))
        if cp_all:
            p = np.concatenate(cp_all)
            g = np.concatenate(cg_all)
            d = np.concatenate(ct_all)
            al, co, cl, cu, errs = compute_calibration_torch(
                torch.tensor(p), torch.tensor(g), torch.tensor(d), conf=10)
            calibration[k] = {
                "alphas": al.tolist(), "coverages": co.tolist(),
                "ci_lower": cl.tolist(), "ci_upper": cu.tolist(),
                "errors": errs, "n_pixels": int(len(p)),
            }
        else:
            calibration[k] = {"n_pixels": 0}

    # --- write outputs ---
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "alpha_cali")
    os.makedirs(out_dir, exist_ok=True)

    # train MLP (independent augmix dataloader)
    if train_mlp:
        _train_mlp_independent(uuid, model, device, aug_type, mlp_epochs, mlp_batch, mlp_lr,
                               excl_suffix, mode, out_dir, max_samples, mlp_freq_enc, mlp_excl, mlp_loss)

    result = {
        "metadata": {
            "uuid":             uuid,
            "epsilons":         epsilons,
            "n_images":         n_images,
            "n_alpha_pixels":   n_total,
            "n_total_std_bins": n_ts_bins_eff,
        },
        "robustness": robustness,
        "alpha_table": alpha_table,
        "calibration": calibration,
    }

    # merged base name
    if mode == "augmix":
        base_name = f"alpha_cali_augmix_{aug_type}{excl_suffix}"
    else:
        base_name = f"alpha_cali{excl_suffix}_merged_eps_{eps_name}"

    if not train_mlp:
        json_path = os.path.join(out_dir, f"{base_name}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → {json_path}")

        # CSV
        csv_rows = []
        for key in ["x", "y", "z"]:
            for cell in alpha_table[key]["grid"]:
                csv_rows.append(cell)
        if csv_rows:
            csv_path = os.path.join(out_dir, f"{base_name}.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                w.writeheader()
                w.writerows(csv_rows)
            print(f"  → {csv_path}")

    # ── per-epsilon alpha tables (FGSM only) ──
    if mode == "fgsm" and not train_mlp:
        for eps in epsilons:
            k = str(eps)
            if not calib_preds[k][0]:
                continue
            # extract per-epsilon pixels
            ep_mp, ep_mg, ep_mt = [], [], []
            for ax in range(3):
                arrs = calib_preds[k][ax]
                if arrs:
                    ep_mp.append(np.concatenate([np.atleast_1d(x) for x in arrs]))
                    ep_mg.append(np.concatenate([np.atleast_1d(x) for x in calib_gts[k][ax]]))
                    ep_mt.append(np.concatenate([np.atleast_1d(x) for x in calib_ts[k][ax]]))
                else:
                    ep_mp.append(np.array([]))
                    ep_mg.append(np.array([]))
                    ep_mt.append(np.array([]))

            # apply exclusion if active
            if excl_active:
                for ax in range(3):
                    keep = np.abs(ep_mp[ax] - excl_centers[ax]) >= excl_radii[ax]
                    ep_mp[ax] = ep_mp[ax][keep]
                    ep_mg[ax] = ep_mg[ax][keep]
                    ep_mt[ax] = ep_mt[ax][keep]

            if len(ep_mp[0]) < 10:
                continue

            # build alpha table
            ep_table = {}
            for ax_idx, key in enumerate(["x", "y", "z"]):
                mp = ep_mp[ax_idx]; mg = ep_mg[ax_idx]; mt = ep_mt[ax_idx]
                if len(mp) < 10:
                    ep_table[key] = {"grid": [], "total_std_edges": [], "pred_edges": []}
                    continue
                valid = np.abs(mp) >= 1e-3
                if valid.sum() < 10:
                    ep_table[key] = {"grid": [], "total_std_edges": [], "pred_edges": []}
                    continue
                pv, gv, tv = mp[valid], mg[valid], mt[valid]
                alpha_v = (gv - pv) / (pv * tv + 1e-12)

                gr_ax = gr[key]
                if excl_active:
                    p_edges = _build_excl_pred_edges(gr_ax, excl_centers[ax_idx], excl_radii[ax_idx], excl_n_pred_bins)
                else:
                    inner = np.linspace(gr_ax[0], gr_ax[1], n_pred_bins + 1)
                    p_edges = np.concatenate([[-np.inf], inner, [np.inf]])

                grid = []
                for i, (tlo, thi) in enumerate(zip(ts_edges[:-1], ts_edges[1:])):
                    m_t = (tv >= tlo) & (tv < thi) if thi == np.inf else (tv >= tlo)
                    for j, (plo, phi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
                        m_p = (pv >= plo) & (pv < phi)
                        m = m_t & m_p
                        if m.sum() < 5:
                            continue
                        a = alpha_v[m]
                        if m.sum() >= 10:
                            lo_a, hi_a = np.percentile(a, [trim_pct * 100, (1 - trim_pct) * 100])
                            a_ma = a[(a >= lo_a) & (a <= hi_a)]
                        else:
                            a_ma = a
                        grid.append({
                            "total_std_bin": i, "total_std_lo": float(tlo), "total_std_hi": float(thi),
                            "pred_bin": j, "pred_lo": float(plo), "pred_hi": float(phi),
                            "n": int(m.sum()),
                            "mean_alpha": float(a_ma.mean()), "std_alpha": float(a.std()),
                            "mean_GT": float(gv[m].mean()),
                            "mean_pred": float(pv[m].mean()),
                            "mean_total_std": float(tv[m].mean()),
                            "axis": key,
                            "eps": k,
                        })
                ep_table[key] = {"grid": grid, "total_std_edges": [float(e) for e in ts_edges],
                                 "pred_edges": [float(e) for e in p_edges]}

            # write CSV
            ep_rows = []
            for key in ["x", "y", "z"]:
                ep_rows.extend(ep_table[key]["grid"])
            if ep_rows:
                ep_csv = os.path.join(out_dir, f"alpha_cali{excl_suffix}_eps_{k}.csv")
                with open(ep_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=ep_rows[0].keys())
                    w.writeheader()
                    w.writerows(ep_rows)
                print(f"  → {ep_csv} ({len(ep_rows)} cells)")

    if not train_mlp:
        # calibration curve
        curve_path = os.path.join(out_dir, "calibration_curve.png")
        _plot_calibration_curve(calibration, curve_path)
        print(f"  → {curve_path}")

        # ── supplementary analyses ──
        if excl_active:
            c_pred_ax = analysis_p_ax
            c_gt_ax   = analysis_g_ax
            c_ts_ax   = analysis_t_ax
        else:
            c_pred_ax = [np.concatenate([np.atleast_1d(x) for x in merged_p[ax]]) for ax in range(3)]
            c_gt_ax   = [np.concatenate([np.atleast_1d(x) for x in merged_g[ax]]) for ax in range(3)]
            c_ts_ax   = [np.concatenate([np.atleast_1d(x) for x in merged_t[ax]]) for ax in range(3)]

        # clean
        # pred edges from main alpha table (used also for mini tables)
        pred_edge_map = {k: alpha_table[k]["pred_edges"] for k in ["x", "y", "z"]}

        tbl_clean = _build_mini_alpha_table(c_pred_ax, c_gt_ax, c_ts_ax, gr, step, std_bins, trim_pct,
                                             pred_edges_dict=pred_edge_map)
        _plot_alpha_correct_vs_unc_for(c_pred_ax, c_gt_ax, c_ts_ax, tbl_clean, n_ts_bins_eff,
                                       os.path.join(out_dir, "alpha_correct_vs_unc.png"))
        _plot_ct_vs_uncertainty_for(c_pred_ax, c_gt_ax, c_ts_ax, gr, step, n_ts_bins,
                                    os.path.join(out_dir, "ct_vs_uncertainty.png"))
        _plot_gt_conditioned_for(c_pred_ax, c_gt_ax,
                                 os.path.join(out_dir, "gt_conditioned.png"))

        # per epsilon
        for eps in epsilons:
            k = str(eps)
            if calib_counts[k][0] == 0:
                continue
            e_pred_ax = [np.concatenate([np.atleast_1d(x) for x in calib_preds[k][ax]]) for ax in range(3)]
            e_gt_ax   = [np.concatenate([np.atleast_1d(x) for x in calib_gts[k][ax]])   for ax in range(3)]
            e_ts_ax   = [np.concatenate([np.atleast_1d(x) for x in calib_ts[k][ax]])    for ax in range(3)]
            tbl_e = _build_mini_alpha_table(e_pred_ax, e_gt_ax, e_ts_ax, gr, step, std_bins, trim_pct,
                                             pred_edges_dict=pred_edge_map)
            tag = f"eps_{k}"
            _plot_alpha_correct_vs_unc_for(e_pred_ax, e_gt_ax, e_ts_ax, tbl_e, n_ts_bins,
                                           os.path.join(out_dir, f"alpha_correct_vs_unc_{tag}.png"))
            _plot_ct_vs_uncertainty_for(e_pred_ax, e_gt_ax, e_ts_ax, gr, step, n_ts_bins,
                                        os.path.join(out_dir, f"ct_vs_uncertainty_{tag}.png"))
            _plot_gt_conditioned_for(e_pred_ax, e_gt_ax,
                                     os.path.join(out_dir, f"gt_conditioned_{tag}.png"))


    # ── supplementary analyses (shared across clean + per-ε) ─────────────────────

def _build_mini_alpha_table(pred_by_ax, gt_by_ax, ts_by_ax, gr, step, std_bins, trim_pct, pred_edges_dict=None):
    """Build per-axis alpha table from (pred, gt, ts) arrays. Returns {ax: lookup_dict}."""
    ts_edges = np.array([0.0] + list(std_bins) + [np.inf])
    alpha_tbl = {}
    for ax_idx, key in enumerate(["x", "y", "z"]):
        p = pred_by_ax[ax_idx]
        g = gt_by_ax[ax_idx]
        t = ts_by_ax[ax_idx]
        valid = np.abs(p) >= 1e-3
        if valid.sum() < 10:
            alpha_tbl[key] = {"edges_ts": [], "edges_pred": [], "grid": {}}
            continue
        pv, gv, tv = p[valid], g[valid], t[valid]
        alpha_v = (gv - pv) / (pv * tv + 1e-12)
        if pred_edges_dict and key in pred_edges_dict:
            p_edges = pred_edges_dict[key]
        else:
            gr_ax = gr[key]
            inner = np.linspace(gr_ax[0], gr_ax[1], n_pred_bins + 1)
            p_edges = np.concatenate([[-np.inf], inner, [np.inf]])

        grid = {}
        for i, (tlo, thi) in enumerate(zip(ts_edges[:-1], ts_edges[1:])):
            m_t = (tv >= tlo) & (tv < thi)
            for j, (plo, phi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
                m = m_t & (pv >= plo) & (pv < phi)
                n = m.sum()
                if n < 5:
                    continue
                a = alpha_v[m]
                if n >= 10:
                    lo_a, hi_a = np.percentile(a, [trim_pct * 100, (1 - trim_pct) * 100])
                    a = a[(a >= lo_a) & (a <= hi_a)]
                grid[(i, j)] = float(a.mean())
        alpha_tbl[key] = {"edges_ts": ts_edges, "edges_pred": p_edges, "grid": grid}
    return alpha_tbl


def _correct_with_alpha(pred_by_ax, ts_by_ax, alpha_tbl):
    """Apply alpha lookup correction per axis. Returns corrected [3][N] arrays."""
    corrected = [p.copy() for p in pred_by_ax]
    for ax_idx, key in enumerate(["x", "y", "z"]):
        tbl = alpha_tbl[key]
        if len(tbl["edges_ts"]) == 0 or len(tbl["edges_pred"]) == 0:
            continue
        p = pred_by_ax[ax_idx]
        t = ts_by_ax[ax_idx]
        ts_edges = tbl["edges_ts"]
        p_edges = tbl["edges_pred"]
        grid = tbl["grid"]
        for i, (tlo, thi) in enumerate(zip(ts_edges[:-1], ts_edges[1:])):
            for j, (plo, phi) in enumerate(zip(p_edges[:-1], p_edges[1:])):
                ma = grid.get((i, j))
                if ma is None:
                    continue
                m = (t >= tlo) & (t < thi) & (p >= plo) & (p < phi)
                corrected[ax_idx][m] = p[m] + ma * p[m] * t[m]
    return corrected


def _plot_alpha_correct_vs_unc_for(pred_by_ax, gt_by_ax, ts_by_ax, alpha_tbl, n_ts_bins, save_path_base, tag=""):
    """DER vs H MAE per total_std bin (percentile + value panels)."""
    # scalar total_std per pixel: sqrt(ts_x² + ts_y² + ts_z²)
    ts_scalar = np.sqrt(ts_by_ax[0]**2 + ts_by_ax[1]**2 + ts_by_ax[2]**2)
    ts_edges = np.percentile(ts_scalar, np.linspace(0, 100, n_ts_bins + 1))
    pct_c = [(i + 0.5) * (100.0 / n_ts_bins) for i in range(n_ts_bins)]
    corrected = _correct_with_alpha(pred_by_ax, ts_by_ax, alpha_tbl)

    der_mae, h_mae, ts_val = [], [], []
    for lo, hi in zip(ts_edges[:-1], ts_edges[1:]):
        m = (ts_scalar >= lo) & (ts_scalar < hi)
        if m.sum() < 10:
            continue
        der_all = np.sqrt(((pred_by_ax[0][m]-gt_by_ax[0][m])**2).astype(float) +
                          ((pred_by_ax[1][m]-gt_by_ax[1][m])**2).astype(float) +
                          ((pred_by_ax[2][m]-gt_by_ax[2][m])**2).astype(float))
        h_all = np.sqrt(((corrected[0][m]-gt_by_ax[0][m])**2).astype(float) +
                        ((corrected[1][m]-gt_by_ax[1][m])**2).astype(float) +
                        ((corrected[2][m]-gt_by_ax[2][m])**2).astype(float))
        ts_val.append(float(ts_scalar[m].mean()))
        der_mae.append(float(der_all.mean()))
        h_mae.append(float(h_all.mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(pct_c, der_mae, "bo-", markersize=5, label="DER")
    ax1.plot(pct_c, h_mae,  "mo-", markersize=5, label="H (alpha-correct)")
    ax1.set_xlabel("total_std percentile"); ax1.set_ylabel("MAE"); ax1.legend(); ax1.grid(True, alpha=0.2); ax1.set_xlim(0,100)
    ax2.plot(ts_val, der_mae, "bo-", markersize=5, label="DER")
    ax2.plot(ts_val, h_mae,  "mo-", markersize=5, label="H (alpha-correct)")
    ax2.set_xlabel("mean total_std"); ax2.set_ylabel("MAE"); ax2.set_xscale("log"); ax2.legend(); ax2.grid(True, alpha=0.2)
    fig.suptitle(f"alpha-correct vs uncertainty {tag}")
    plt.tight_layout()
    plt.savefig(save_path_base + ("_" + tag if tag else "") + ".png" if "png" not in save_path_base else save_path_base, dpi=200)
    plt.close()


def _plot_ct_vs_uncertainty_for(pred_by_ax, gt_by_ax, ts_by_ax, gr, step, n_ts_bins, save_path):
    """Per GT bin: total_std vs bias curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax_i, (ax, key) in enumerate(zip(axes, ["x", "y", "z"])):
        p, g, t = pred_by_ax[ax_i], gt_by_ax[ax_i], ts_by_ax[ax_i]
        gr_ax = gr[key]
        gt_edges = np.arange(gr_ax[0], gr_ax[1] + step * 0.5, step)
        n_gt = len(gt_edges) - 1
        picks = [0, n_gt//4, n_gt//2, 3*n_gt//4, n_gt-1] if n_gt > 1 else [0]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(picks)))
        for pi, gi in enumerate(picks):
            if gi >= n_gt: continue
            lo, hi = gt_edges[gi], gt_edges[gi+1]
            mg = (g >= lo) & (g < hi)
            if mg.sum() < 30: continue
            tg, pg, gg = t[mg], p[mg], g[mg]
            ts_edges = np.percentile(tg, np.linspace(0, 100, n_ts_bins+1))
            xs, ys = [], []
            for tlo, thi in zip(ts_edges[:-1], ts_edges[1:]):
                mc = (tg >= tlo) & (tg < thi)
                if mc.sum() < 5: continue
                xs.append(float(tg[mc].mean()))
                ys.append(float((pg[mc] - gg[mc]).mean()))
            gt_mid = float(gg.mean())
            ax.plot(xs, ys, "o-", color=colors[pi], markersize=3, linewidth=1, label=f"GT≈{gt_mid:+.2f}")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
        ax.set_title(f"GT_{key}"); ax.set_xscale("log"); ax.legend(fontsize=6); ax.grid(True, alpha=0.15)
    fig.suptitle("ct vs uncertainty")
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


def _plot_gt_conditioned_for(pred_by_ax, gt_by_ax, save_path):
    """Mean pred + p25-p75 vs GT, 3 panels with identity line + degeneration center."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax_i, (ax, name) in enumerate(zip(axes, ["x", "y", "z"])):
        g = gt_by_ax[ax_i]; p = pred_by_ax[ax_i]
        gt_edges = np.percentile(g, np.linspace(0, 100, 21))
        gtm, ma, p25, p75 = [], [], [], []
        for lo, hi in zip(gt_edges[:-1], gt_edges[1:]):
            m = (g >= lo) & (g < hi)
            if m.sum() < 3: continue
            gtm.append(float(g[m].mean()))
            ma.append(float(p[m].mean())); p25.append(float(np.percentile(p[m],25))); p75.append(float(np.percentile(p[m],75)))
        ax.fill_between(gtm, p25, p75, color="#1f77b4", alpha=0.12)
        ax.plot(gtm, ma, "o-", color="#1f77b4", markersize=3, label="DER")
        mx = max(abs(np.array(gtm).min()), abs(np.array(gtm).max())) * 1.1
        ax.plot([-mx, mx], [-mx, mx], "gray", linestyle=":", alpha=0.5)
        # degeneration center: global mean prediction
        dc = float(p.mean())
        ax.axhline(dc, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(mx * 0.95, dc, f"{dc:+.1e}", color="red", fontsize=7, va="bottom", ha="right")
        ax.set_title(f"GT_{name}"); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    fig.suptitle("prediction vs GT")
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha calibration via FGSM on validation")
    parser.add_argument("--uuid", default="e24d72fb-b4d7-4c40-b00a-a4aa2f8213e8",
                        help="DER model UUID")
    parser.add_argument("--epsilons", nargs="*", type=float,
                        default=None, help="FGSM epsilon values")
    parser.add_argument("--n_ts_bins", type=int, default=10,
                        help="Number of total_std percentile bins (for plots only)")
    parser.add_argument("--std_bins", nargs="*", type=float,
                        default=None, help="Fixed total_std edges for alpha table")
    parser.add_argument("--max_samples", type=int, default=5,
                        help="Max validation images")
    parser.add_argument("--excl_cx", type=float, default=0.045, help="Exclusion center X")
    parser.add_argument("--excl_cy", type=float, default=0.057, help="Exclusion center Y")
    parser.add_argument("--excl_cz", type=float, default=0.16, help="Exclusion center Z")
    parser.add_argument("--excl_rx", type=float, default=0.05, help="Exclusion radius X")
    parser.add_argument("--excl_ry", type=float, default=0.05, help="Exclusion radius Y")
    parser.add_argument("--excl_rz", type=float, default=0.05, help="Exclusion radius Z")
    parser.add_argument("--mode", choices=["fgsm", "augmix"], default="fgsm")
    parser.add_argument("--aug_type", default=None, help="SpaceAugTransform aug_type (default: from cfg.yaml)")
    parser.add_argument("--train_mlp", action="store_true", help="Train AlphaMLP after calibration")
    parser.add_argument("--mlp_epochs", type=int, default=1000, help="MLP training epochs")
    parser.add_argument("--mlp_lr", type=float, default=1e-3, help="MLP learning rate")
    parser.add_argument("--mlp_batch", type=int, default=16, help="MLP batch size (images per batch)")
    parser.add_argument("--mlp_raw", action="store_true", default=False, help="Use raw features (no freq encoding) in MLP")
    parser.add_argument("--mlp_keep_center", action="store_true", default=False, help="Keep center pixels in MLP training (no exclusion)")
    parser.add_argument("--mlp_loss", choices=["l1", "l2", "smooth_l1"], default="l2",
                        help="Loss function for MLP training")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    epsilons = args.epsilons if args.epsilons else sample_epsilon(None)
    print(f"Epsilons: {epsilons}")
    calibrate(args.uuid, epsilons, args.n_ts_bins, args.max_samples, args.std_bins, args.device,
              args.excl_cx, args.excl_cy, args.excl_cz, args.excl_rx, args.excl_ry, args.excl_rz,
              args.mode, args.aug_type,
              args.train_mlp, args.mlp_epochs, args.mlp_lr, args.mlp_batch,
              not args.mlp_raw, not args.mlp_keep_center, args.mlp_loss)
