"""Stable Learning (StableNet) components.

Implements sample reweighting via Random Fourier Features (RFF) with
global balancing, following "StableNet: Stable Learning via Sample
Reweighting" (Zhang et al., ICML 2021).

Components:
  - FeatureNet:   small MLP projecting backbone features to low-dim z
  - RFFLayer:     fixed random Fourier features u = sqrt(2/m) * cos(z W + b)
  - StableNetState: global balancing buffers B_z / B_u + decorrelation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    """Project backbone features z0 to low-dim z (trained jointly with task)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, z0):
        return self.net(z0)


class RFFLayer(nn.Module):
    """Random Fourier Features with fixed random weights.

    u(z) = sqrt(2/m) * cos(z @ W + b),  W ~ N(0, sigma^2), b ~ U(0, 2*pi)
    """

    def __init__(self, in_dim: int, rff_dim: int, sigma: float = 1.0):
        super().__init__()
        self.register_buffer(
            "W", torch.randn(in_dim, rff_dim) * sigma, persistent=False)
        self.register_buffer(
            "b", torch.rand(rff_dim) * 2.0 * 3.141592653589793, persistent=False)
        self.scale = (2.0 / rff_dim) ** 0.5

    def forward(self, z):
        return self.scale * torch.cos(z @ self.W + self.b)


class StableNetState:
    """Global balancing buffers + decorrelation loss (StableNet).

    B_z, B_u are global weighted-mean buffers updated as
        B <- (1 - beta) * B + beta * mean(w * f)
    The independence loss uses global-centered cross-covariance:
        L = sum_{j,k} cov(z_j, u_k)^2
    """

    def __init__(self, n_z: int, rff_dim: int, beta: float = 0.9, device="cuda:0"):
        self.n_z = n_z
        self.rff_dim = rff_dim
        self.beta = beta
        self.device = device
        self.B_z = torch.zeros(1, n_z, device=device)
        self.B_u = torch.zeros(1, rff_dim, device=device)

    def loss(self, w, z, u):
        """Decorrelation loss given sample weights w [B], features z [B,n_z],
        rff u [B,m]. Uses global buffers B_z/B_u for centering."""
        w = w.clamp(min=0.0)
        w = w / (w.mean() + 1e-8)  # normalize mean 1
        n = z.shape[0]
        z_w = (z * w[:, None] - self.B_z)   # [B, n_z]
        u_w = (u * w[:, None] - self.B_u)   # [B, m]
        cov = (z_w.T @ u_w) / n             # [n_z, m]
        return (cov ** 2).sum()

    def update_global_B(self, w, z, u):
        """Update global buffers with final (detached) weights."""
        w = w.clamp(min=0.0)
        w = w / (w.mean() + 1e-8)
        with torch.no_grad():
            self.B_z = (1 - self.beta) * self.B_z + self.beta * (w[:, None] * z).mean(0, keepdim=True)
            self.B_u = (1 - self.beta) * self.B_u + self.beta * (w[:, None] * u).mean(0, keepdim=True)
