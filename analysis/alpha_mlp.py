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

    # build per-axis per-image arrays
    eps = 1e-3
    models = {"x": AlphaMLPSingle().to(device),
              "y": AlphaMLPSingle().to(device),
              "z": AlphaMLPSingle().to(device)}

    # build image-indexed data per axis
    axis_images = {"x": [], "y": [], "z": []}
    for k in epsilons:
        k = str(k)
        n_imgs = len(calib_preds[k][0])
        for i in range(n_imgs):
            for ax_idx, ax_name in enumerate(["x", "y", "z"]):
                p = np.atleast_1d(calib_preds[k][ax_idx][i]).ravel()
                g = np.atleast_1d(calib_gts[k][ax_idx][i]).ravel()
                t = np.atleast_1d(calib_ts[k][ax_idx][i]).ravel()
                valid = np.abs(p) >= eps
                if valid.sum() > 0:
                    axis_images[ax_name].append((p[valid], g[valid], t[valid]))

    n_imgs_x = len(axis_images["x"])
    print(f"  MLP training: {n_imgs_x} images/axis, {epochs} epochs, lr={lr}, batch={batch_size} images")

    for ax_name, model in models.items():
        imgs = axis_images[ax_name]
        n_imgs = len(imgs)
        n_train = int(0.8 * n_imgs)
        n_val = n_imgs - n_train

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # prepare val set (all val images at once)
        val_pred = torch.cat([torch.tensor(imgs[i][0], dtype=torch.float32) for i in range(n_train, n_imgs)]).to(device)
        val_gt   = torch.cat([torch.tensor(imgs[i][1], dtype=torch.float32) for i in range(n_train, n_imgs)]).to(device)
        val_ts   = torch.cat([torch.tensor(imgs[i][2], dtype=torch.float32) for i in range(n_train, n_imgs)]).to(device)
        alpha_val = (val_gt - val_pred) / (val_pred * val_ts + 1e-12)

        best_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n_train)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                idx = perm[start:end].tolist()

                # concat selected images
                pred_b = torch.cat([torch.tensor(imgs[i][0], dtype=torch.float32) for i in idx]).to(device)
                gt_b   = torch.cat([torch.tensor(imgs[i][1], dtype=torch.float32) for i in idx]).to(device)
                ts_b   = torch.cat([torch.tensor(imgs[i][2], dtype=torch.float32) for i in idx]).to(device)
                alpha_b = (gt_b - pred_b) / (pred_b * ts_b + 1e-12)

                pred_out = model(ts_b, pred_b)
                loss = nn.functional.mse_loss(pred_out, alpha_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_out = model(val_ts, val_pred)
                val_loss = nn.functional.mse_loss(val_out, alpha_val).item()

            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"    {ax_name} epoch {epoch:3d}: train_loss={total_loss/n_batches:.4f} val_loss={val_loss:.4f}")
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
