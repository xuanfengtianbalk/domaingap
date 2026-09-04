"""Stable Learning (StableNet) components — aligned with the official repo
(https://github.com/xxgege/StableNet, CVPR 2021).

Ports:
  loss_reweighting.py          → RFFTransform / weighted_cov / lossb
  training/reweighting.py      → weight_learner
  models/resnet_with_table.py  → StableNetState (pre_features / pre_weight1 table)

Core idea: learn per-sample weights (softmax simplex) that decorrelate
RFF-transformed features (feature-feature off-diagonal covariance), using a
global table of pre-saved features/weights (EMA, presave_ratio).
"""

import torch
import torch.nn as nn


class RFFTransform(nn.Module):
    """Official random_fourier_features_gpu.

    W ~ randn(num_f, 1) / sigma   (one scalar per fourier space)
    b ~ U(0, 2*pi)  shape [d, num_f]
    mid = x @ W.t() + b  → [n, d, num_f]
    mid normalized to [0, pi/2] along the feature dim d
    Z = sqrt(2/num_f) * (cos(mid) + sin(mid))   (sum=True in official)
    """

    def __init__(self, feature_dim: int, num_f: int = 1, sigma: float = 1.0):
        super().__init__()
        self.num_f = num_f
        self.feature_dim = feature_dim
        self.register_buffer(
            "W", torch.randn(num_f, 1) / sigma, persistent=False)
        self.register_buffer(
            "b", 2.0 * 3.141592653589793 * torch.rand(feature_dim, num_f),
            persistent=False)
        self.scale = (2.0 / num_f) ** 0.5

    def forward(self, x):
        """x: [n, d] → [n, d, num_f]"""
        n, d = x.shape
        x = x.view(n, d, 1)
        mid = x @ self.W.t() + self.b.unsqueeze(0)      # [n, d, num_f]
        mid = mid - mid.min(dim=1, keepdim=True)[0]
        mid = mid / (mid.max(dim=1, keepdim=True)[0] + 1e-12)
        mid = mid * (3.141592653589793 / 2.0)
        return self.scale * (torch.cos(mid) + torch.sin(mid))


def weighted_cov(x, sw):
    """Official cov(): weighted second moment minus outer product of
    weighted means. x: [n, d], sw: [n] (softmax). Returns [d, d]."""
    sw = sw.view(-1, 1)
    cov = (sw * x).t() @ x
    e = (sw * x).sum(0).view(-1, 1)
    return cov - e @ e.t()


class StableNetState:
    """Global table of pre-saved features and weights (Eq. 11/14).

    pre_features: [n_feature, d] init zeros
    pre_weight1:  [n_feature, 1] init ones
    n_feature should equal the batch size (official default 128 == bs 128).
    """

    def __init__(self, n_feature: int, feature_dim: int,
                 presave_ratio: float = 0.9, device="cuda:0"):
        self.n_feature = n_feature
        self.feature_dim = feature_dim
        self.presave_ratio = presave_ratio
        self.device = device
        self.pre_features = torch.zeros(n_feature, feature_dim, device=device)
        self.pre_weight1 = torch.ones(n_feature, 1, device=device)
        self._first_batch_count = 0

    def lossb(self, all_feature, sw_all, rff):
        """Official lossb_expect: per-fourier-space weighted feature-feature
        covariance, penalizing off-diagonal (inter-dimension) terms."""
        z = rff(all_feature)                              # [N, d, num_f]
        loss = torch.zeros((), device=self.device)
        for i in range(z.size(-1)):
            cov1 = weighted_cov(z[:, :, i], sw_all)       # [d, d]
            loss = loss + (cov1 ** 2).sum() - (cov1 ** 2).trace()
        return loss

    def lossp(self, sw, decay_pow):
        """Official lossp: sum of softmax(weight)^decay_pow."""
        return sw.pow(decay_pow).sum()

    def update(self, cfeatures, weight_final, global_epoch, batch_idx):
        """Eq. 14: first 10 iterations of epoch 0 → running average;
        otherwise EMA blend with presave_ratio (0.9 old, 0.1 new)."""
        with torch.no_grad():
            if global_epoch == 0 and batch_idx < 10:
                self.pre_features = (self.pre_features * batch_idx + cfeatures) / (batch_idx + 1)
                self.pre_weight1 = (self.pre_weight1 * batch_idx + weight_final) / (batch_idx + 1)
            elif cfeatures.shape[0] < self.pre_features.shape[0]:
                B = cfeatures.shape[0]
                self.pre_features[:B] = (self.pre_features[:B] * self.presave_ratio
                                         + cfeatures * (1 - self.presave_ratio))
                self.pre_weight1[:B] = (self.pre_weight1[:B] * self.presave_ratio
                                        + weight_final * (1 - self.presave_ratio))
            else:
                self.pre_features = (self.pre_features * self.presave_ratio
                                     + cfeatures * (1 - self.presave_ratio))
                self.pre_weight1 = (self.pre_weight1 * self.presave_ratio
                                    + weight_final * (1 - self.presave_ratio))


def weight_learner(cfeatures, state, rff, num_f, epochb, lrbl, lambdap, decay_pow,
                   global_epoch, batch_idx, first_step_cons,
                   lambda_decay_rate, lambda_decay_epoch, min_lambda_times):
    """Official weight_learner (training/reweighting.py).

    Returns (softmax(weight).detach(), state).
    """
    B = cfeatures.shape[0]
    all_feature = torch.cat([cfeatures, state.pre_features.detach()], dim=0)  # Eq.11

    weight = torch.ones(B, 1, device=state.device, requires_grad=True)
    momentum_buf = None
    for _ in range(epochb):
        all_weight = torch.cat([weight, state.pre_weight1.detach()], dim=0)
        sw_all = torch.softmax(all_weight, dim=0)
        sw_local = torch.softmax(weight, dim=0)

        lossb = state.lossb(all_feature, sw_all, rff)
        lossp = state.lossp(sw_local, decay_pow)
        lambdap_eff = lambdap * max(
            lambda_decay_rate ** (global_epoch // lambda_decay_epoch),
            min_lambda_times)
        lossg = lossb / lambdap_eff + lossp
        if global_epoch == 0:
            lossg = lossg * first_step_cons

        grad = torch.autograd.grad(lossg, weight)[0]
        if momentum_buf is None:
            momentum_buf = torch.zeros_like(grad)
        momentum_buf = 0.9 * momentum_buf + grad
        weight = (weight.detach() - lrbl * momentum_buf).requires_grad_(True)

    softmax_weight = torch.softmax(weight, dim=0).detach()
    return softmax_weight, state
