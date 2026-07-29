"""alpha_mlp.py — Unified 3-axis coordinate correction model.

Input:  (ts_x, pred_x, ts_y, pred_y, ts_z, pred_z) × 6 FreqEnc(10) → 120D
MLP:    120 → 256 → 128 → 64 → 3
Output: (corrected_x, corrected_y, corrected_z)

Directly predicts corrected 3D coordinates from uncertainty + raw prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import math


class FreqEmbed(nn.Module):
    """Positional encoding: (B,1) → (B, 2*freqs)."""
    def __init__(self, freqs=10):
        super().__init__()
        self.freqs = freqs

    def forward(self, x):
        out = []
        for i in range(self.freqs):
            f = 2.0 ** i
            out.append(torch.sin(f * math.pi * x))
            out.append(torch.cos(f * math.pi * x))
        return torch.cat(out, dim=-1)


class UnifiedCorrectionMLP(nn.Module):
    """Feature-mode MLP → 3 (corrected xyz). Optional freq encoding."""

    def __init__(self, use_freq_enc: bool = True, feature_mode: str = "all"):
        super().__init__()
        self.use_freq_enc = use_freq_enc
        self.feature_mode = feature_mode
        if feature_mode == "ts_only":
            n_feat = 3
        else:
            n_feat = 6
        in_dim = n_feat * (2 * 10) if use_freq_enc else n_feat
        if use_freq_enc:
            self.embed = FreqEmbed(10)
        else:
            self.embed = None
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 3),
        )

    def forward(self, tx, px, ty, py, tz, pz):
        if self.feature_mode == "ts_only":
            feat_list = [tx, ty, tz]
        else:
            feat_list = [tx, px, ty, py, tz, pz]

        if self.use_freq_enc:
            feat = torch.cat([self.embed(f) for f in feat_list], dim=-1)
        else:
            feat = torch.cat(feat_list, dim=-1)
        return self.mlp(feat)


def train_correction_mlp(calib_preds, calib_gts, calib_ts, epsilons,
                         device="cuda:0", batch_size=16, epochs=100, lr=1e-3):
    """Train unified 3-axis correction model.

    Loss = MSE(predicted_corrected_3d, GT_3d) on pixels where all 3 axes
    have |pred_ax| >= 1e-3.

    Returns state_dict of the trained model.
    """
    eps = 1e-3
    model = UnifiedCorrectionMLP().to(device)

    # build aligned per-image arrays across axes
    aligned_imgs = []
    for k in [str(e) for e in epsilons]:
        n_imgs = len(calib_preds[k][0])
        for i in range(n_imgs):
            px = np.atleast_1d(calib_preds[k][0][i]).ravel()
            py = np.atleast_1d(calib_preds[k][1][i]).ravel()
            pz = np.atleast_1d(calib_preds[k][2][i]).ravel()
            gx = np.atleast_1d(calib_gts[k][0][i]).ravel()
            gy = np.atleast_1d(calib_gts[k][1][i]).ravel()
            gz = np.atleast_1d(calib_gts[k][2][i]).ravel()
            tx = np.atleast_1d(calib_ts[k][0][i]).ravel()
            ty = np.atleast_1d(calib_ts[k][1][i]).ravel()
            tz = np.atleast_1d(calib_ts[k][2][i]).ravel()

            valid = (np.abs(px) >= eps) & (np.abs(py) >= eps) & (np.abs(pz) >= eps)
            if valid.sum() > 0:
                aligned_imgs.append({
                    "tx": tx[valid], "px": px[valid],
                    "ty": ty[valid], "py": py[valid],
                    "tz": tz[valid], "pz": pz[valid],
                    "gx": gx[valid], "gy": gy[valid], "gz": gz[valid],
                })

    n_imgs = len(aligned_imgs)
    n_train = int(0.8 * n_imgs)
    print(f"  MLP training: {n_imgs} images, {epochs} epochs, lr={lr}, batch={batch_size}")

    # val set
    val_ts = {k: torch.cat([torch.tensor(aligned_imgs[i][k], dtype=torch.float32)
                            for i in range(n_train, n_imgs)]).reshape(-1, 1).to(device)
              for k in ["tx", "px", "ty", "py", "tz", "pz"]}
    val_gt = torch.stack([
        torch.cat([torch.tensor(aligned_imgs[i]["gx"], dtype=torch.float32) for i in range(n_train, n_imgs)]),
        torch.cat([torch.tensor(aligned_imgs[i]["gy"], dtype=torch.float32) for i in range(n_train, n_imgs)]),
        torch.cat([torch.tensor(aligned_imgs[i]["gz"], dtype=torch.float32) for i in range(n_train, n_imgs)]),
    ], dim=-1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss, n_batches = 0.0, 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end].tolist()

            ts = {k: torch.cat([torch.tensor(aligned_imgs[i][k], dtype=torch.float32)
                                for i in idx]).reshape(-1, 1).to(device)
                  for k in ["tx", "px", "ty", "py", "tz", "pz"]}
            gt_b = torch.stack([
                torch.cat([torch.tensor(aligned_imgs[i]["gx"], dtype=torch.float32) for i in idx]),
                torch.cat([torch.tensor(aligned_imgs[i]["gy"], dtype=torch.float32) for i in idx]),
                torch.cat([torch.tensor(aligned_imgs[i]["gz"], dtype=torch.float32) for i in idx]),
            ], dim=-1).to(device)

            pred = model(ts["tx"], ts["px"], ts["ty"], ts["py"], ts["tz"], ts["pz"])
            loss = nn.functional.mse_loss(pred, gt_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_ts["tx"], val_ts["px"], val_ts["ty"], val_ts["py"], val_ts["tz"], val_ts["pz"])
            val_loss = nn.functional.mse_loss(val_pred, val_gt).item()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:3d}: train={total_loss/n_batches:.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss

    print(f"    best val_loss={best_val:.4f}")
    return model.state_dict()


def load_correction_mlp(pt_path: str, device: str = "cuda:0"):
    state = torch.load(pt_path, map_location=device, weights_only=True)
    w0 = state["mlp.0.weight"]  # (256, in_dim)
    in_dim = w0.shape[1]
    use_freq = in_dim in (60, 120)
    ts_only = in_dim in (3, 60)
    model = UnifiedCorrectionMLP(use_freq_enc=use_freq, feature_mode="ts_only" if ts_only else "all")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
