"""
MixStyle from: Zhou et al. "Domain Generalization with MixStyle" (ICLR 2021)
https://github.com/KaiyangZhou/mixstyle-release
"""
import torch
import torch.nn as nn
import numpy as np


class MixStyle(nn.Module):
    """MixStyle: mix instance-level feature statistics across batch samples."""
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training:
            return x
        if np.random.rand() > self.p:
            return x

        B = x.size(0)
        # Instance-level mean and std: [B, C, 1, 1] or [B, N, C]
        mu = x.mean(dim=1, keepdim=True) if x.dim() == 3 else x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=1, keepdim=True) if x.dim() == 3 else x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        # Shuffle
        idx = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[idx], sig[idx]

        # Mix
        mu_mix = mu * (1 - self.alpha) + mu2 * self.alpha
        sig_mix = sig * (1 - self.alpha) + sig2 * self.alpha

        return (x - mu) / sig * sig_mix + mu_mix
