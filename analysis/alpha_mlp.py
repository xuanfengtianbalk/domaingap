"""alpha_mlp.py — Learned Alpha Prediction Network (FreqEnc + MLP).

Three independent MLPs (x/y/z), each maps (total_std, pred) → alpha.
Replaces CSV lookup table with continuous differentiable function.
"""

import torch
import torch.nn as nn
import numpy as np
import math, os


class FreqEmbed(nn.Module):
    """Positional encoding: (B,1) → (B, 2*freqs) using sin/cos."""
    def __init__(self, freqs=10):
        super().__init__()
        self.freqs = freqs

    def forward(self, x):
        # x: (B, 1)
        out = []
        for i in range(self.freqs):
            f = 2.0 ** i
            out.append(torch.sin(f * math.pi * x))
            out.append(torch.cos(f * math.pi * x))
        return torch.cat(out, dim=-1)  # (B, 2*freqs)


class AlphaMLPSingle(nn.Module):
    """MLP for one axis: 40 → 128 → 64 → 32 → 1."""
    def __init__(self):
        super().__init__()
        self.embed = FreqEmbed(10)
        self.mlp = nn.Sequential(
            nn.Linear(40, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, ts, pred):
        # ts, pred: (B,) or (B,1)
        if ts.dim() == 1:
            ts = ts.unsqueeze(-1)
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        x_ts   = self.embed(ts)      # (B, 20)
        x_pred = self.embed(pred)    # (B, 20)
        x = torch.cat([x_ts, x_pred], dim=-1)  # (B, 40)
        return self.mlp(x).squeeze(-1)  # (B,)


class AlphaMLP3D(nn.Module):
    """Three independent single-axis MLPs."""
    def __init__(self):
        super().__init__()
        self.x = AlphaMLPSingle()
        self.y = AlphaMLPSingle()
        self.z = AlphaMLPSingle()

    def forward(self, axis, ts, pred):
        if axis == 0 or axis == "x":
            return self.x(ts, pred)
        elif axis == 1 or axis == "y":
            return self.y(ts, pred)
        else:
            return self.z(ts, pred)


def train_alpha_mlp(calib_preds, calib_gts, calib_ts, epsilons,
                    device="cuda:0", batch_size=16, epochs=100, lr=1e-3):
    """Train 3 independent MLPs on per-epsilon raw pixel data.

    Args:
        calib_preds: {eps: [ax0_list, ax1_list, ax2_list]} — per-image arrays
        calib_gts:   same structure
        calib_ts:    same structure
        epsilons:    list of epsilon keys
        device:      torch device
        batch_size:  number of images per batch
        epochs:      training epochs
        lr:          learning rate

    Returns:
        dict of {axis: AlphaMLPSingle} state_dicts
    """
    eps = 1e-3
    models = {"x": AlphaMLPSingle().to(device),
              "y": AlphaMLPSingle().to(device),
              "z": AlphaMLPSingle().to(device)}

    # flatten all data per axis
    all_data = {}
    for ax_idx, ax_name in enumerate(["x", "y", "z"]):
        pred_list, gt_list, ts_list = [], [], []
        for eps_key in epsilons:
            k = str(eps_key)
            for arr in calib_preds[k][ax_idx]:
                pred_list.append(np.atleast_1d(arr))
                gt_list.append(np.atleast_1d(calib_gts[k][ax_idx][0] if len(calib_gts[k][ax_idx]) <= len(pred_list) else calib_gts[k][ax_idx][len(pred_list)-1]))
        # Flatten: calib_preds[k][ax] is a list of per-image arrays
        flat_pred, flat_gt, flat_ts = [], [], []
        for k in epsilons:
            k = str(k)
            for i in range(len(calib_preds[k][ax_idx])):
                p = calib_preds[k][ax_idx][i]
                g = calib_gts[k][ax_idx][i]
                t = calib_ts[k][ax_idx][i]
                flat_pred.append(np.atleast_1d(p).ravel())
                flat_gt.append(np.atleast_1d(g).ravel())
                flat_ts.append(np.atleast_1d(t).ravel())

        all_pred = np.concatenate(flat_pred)
        all_gt   = np.concatenate(flat_gt)
        all_ts   = np.concatenate(flat_ts)

        # filter |pred| < eps
        valid = np.abs(all_pred) >= eps
        all_data[ax_name] = {
            "pred": torch.tensor(all_pred[valid], dtype=torch.float32),
            "gt":   torch.tensor(all_gt[valid], dtype=torch.float32),
            "ts":   torch.tensor(all_ts[valid], dtype=torch.float32),
        }

    n_total = len(all_data["x"]["pred"])
    n_images = sum(len(calib_preds[str(epsilons[0])][0]) for _ in [0]) if epsilons else 0
    print(f"  MLP training: {n_total} pixels, {epochs} epochs, lr={lr}")

    for ax_name, model in models.items():
        data = all_data[ax_name]
        pred_all = data["pred"].to(device)
        gt_all   = data["gt"].to(device)
        ts_all   = data["ts"].to(device)

        alpha_true = (gt_all - pred_all) / (pred_all * ts_all + 1e-12)

        n = len(pred_all)
        idx = torch.randperm(n)
        n_train = int(0.8 * n)
        idx_train, idx_val = idx[:n_train], idx[n_train:]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        best_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n_train)
            total_loss = 0.0
            batches = 0
            # batch by images: approximate by random sampling equal-sized chunks
            chunk_size = 4096
            for start in range(0, n_train, chunk_size):
                end = min(start + chunk_size, n_train)
                idx_b = idx_train[perm[start:end]]
                ts_b   = ts_all[idx_b]
                pred_b = pred_all[idx_b]
                alpha_b = alpha_true[idx_b]

                pred_out = model(ts_b, pred_b)
                loss = nn.functional.mse_loss(pred_out, alpha_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1

            scheduler.step()

            # validation
            model.eval()
            with torch.no_grad():
                val_out = model(ts_all[idx_val], pred_all[idx_val])
                val_loss = nn.functional.mse_loss(val_out, alpha_true[idx_val]).item()

            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"    {ax_name} epoch {epoch:3d}: train_loss={total_loss/batches:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss

        print(f"    {ax_name}: best val_loss={best_loss:.4f}")

    return {"x": models["x"].state_dict(),
            "y": models["y"].state_dict(),
            "z": models["z"].state_dict()}


def load_alpha_mlp(pt_path: str, device: str = "cuda:0"):
    """Load trained MLP model for inference."""
    model3d = AlphaMLP3D()
    state = torch.load(pt_path, map_location=device, weights_only=True)
    model3d.x.load_state_dict(state["x"])
    model3d.y.load_state_dict(state["y"])
    model3d.z.load_state_dict(state["z"])
    model3d.to(device)
    model3d.eval()
    return model3d
