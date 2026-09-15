"""LaSt-ViT (LazyStrike) stability score — paper Eq. 4/5.

Frequency-aware channel stability:
  x_hat = IFFT(Gaussian(FFT(x)))          # channel-dim low-pass
  S[i,j] = x_hat[i,j] / (|x_hat[i,j] - x[i,j]| + eps)   # per-channel stability

Parameter-free (fixed Gaussian kernel). Paper: kernel_size=D, sigma=sqrt(D).
"""

import math

import torch
import torch.nn as nn


class StabilityScore(nn.Module):
    """Channel-wise stability score for patch/cell features [N, D].

    Usage:
        stab = StabilityScore(D, sigma=None)
        S = stab(x)          # [N, D], higher = more low-frequency-stable channel
    """

    def __init__(self, dim: int, sigma: float = None, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        sigma = sigma if sigma is not None else math.sqrt(dim)
        half = dim // 2
        idx = torch.arange(-(half - 1), half + 1).float()
        g = torch.exp(-0.5 * (idx / sigma) ** 2)
        g = g / g.max()
        self.register_buffer("g", g.unsqueeze(0), persistent=False)  # [1, D]

    def forward(self, x):
        """x: [N, D] → S: [N, D] per-channel stability."""
        X = torch.fft.fft(x, dim=-1)
        X_lp = torch.fft.fftshift(X, dim=-1) * self.g
        X_lp = torch.fft.ifftshift(X_lp, dim=-1)
        x_hat = torch.fft.ifft(X_lp, dim=-1).real
        S = x_hat / (torch.abs(x_hat - x) + self.eps)
        return S

    def cell_stability(self, x):
        """cell-level aggregation: mean over channels → [N]."""
        return self(x).mean(dim=-1)

    def vote_count(self, x, K=None):
        """Paper-style channel-wise Top-K vote (Eq. 6/8).

        For each channel j, select the K most stable cells; vote[i] =
        number of channels where cell i is selected. Returns [N].
        """
        S = self(x)                       # [N, D]
        N, D = S.shape
        K = K if K is not None else max(1, N // 2)
        _, idx = torch.topk(S.t(), K, dim=1)   # [D, K] cell indices
        vote = torch.zeros(N, device=x.device, dtype=torch.long)
        vote.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=x.device, dtype=torch.long))
        return vote.float()


def lastvit_context(feat, K=None, eps=1e-6):
    """LaSt-ViT global context (paper Eq. 4-7). Parameter-free & differentiable.

    feat: [B, C, H, W] decoder feature map.
    Returns g: [B, C] — per-channel mean of the K most stable spatial patches.

    Gradients flow through the selected patches back into the feature map,
    which is the paper's training mechanism (anchor the global context to
    foreground-stable patches).
    """
    B, C, H, W = feat.shape
    x = feat.flatten(2).permute(0, 2, 1)              # [B, N, C]
    N = x.shape[1]
    K = K if K is not None else max(1, N // 2)

    X = torch.fft.fft(x, dim=-1)                      # channel-dim FFT
    half = C // 2
    idx = torch.arange(-(half - 1), half + 1, device=feat.device).float()
    gk = torch.exp(-0.5 * (idx / (C ** 0.5)) ** 2)
    gk = gk / gk.max()
    X_lp = torch.fft.fftshift(X, dim=-1) * gk.view(1, 1, -1)
    X_lp = torch.fft.ifftshift(X_lp, dim=-1)
    x_hat = torch.fft.ifft(X_lp, dim=-1).real         # [B, N, C]

    S = x_hat / (torch.abs(x_hat - x) + eps)          # [B, N, C] stability
    _, idx_k = torch.topk(S.permute(0, 2, 1), K, dim=-1)   # [B, C, K]
    sel = torch.gather(x.transpose(1, 2), 2, idx_k)   # [B, C, K]
    return sel.mean(dim=-1)                           # [B, C]
