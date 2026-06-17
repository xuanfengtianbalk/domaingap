"""Deep Evidential Regression for coordinate prediction (Amini et al. NeurIPS 2020).

Models a Normal Inverse Gamma (NIG) distribution over each coordinate output:
  y ~ N(gamma, sigma² / ν)
  sigma² ~ InvGamma(alpha, beta)

Network outputs: c (gamma), logl (log ν), loga (log alpha), logb (log beta)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Negative log-likelihood
# ---------------------------------------------------------------------------

def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    return (t1 - t2 + t3 + t4 - t5).mean()


def gs_l_nll(gamma, _v, alpha, beta, y):
    a_var = beta / (alpha - 1)
    return (0.5 * (2 * torch.pi * a_var).log() + (y - gamma) ** 2 / (2 * a_var)).mean()


def nig_reg(gamma, v, alpha, beta, y):
    return ((y - gamma).abs() * (2 * v + alpha)).mean()


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------

class EvidentialLossSumOfSquares(nn.Module):
    """Deep Evidential Regression loss for coordinate prediction.

    Parameters
    ----------
    lamb : float
        Regularisation weight.
    active : str or callable
        Activation for alpha / beta / nu from raw logits (``'exp'`` or ``'softplus'``).
    use_gs : bool
        If True use Gaussian NLL instead of NIG NLL.
    """

    def __init__(self, lamb=1e-3, active='exp', use_gs=False):
        super().__init__()
        self.lamb = lamb
        self.use_gs = use_gs

        if active == 'exp':
            self.active = torch.exp
        elif active == 'softplus':
            self.active = F.softplus
        else:
            self.active = active

    def forward(self, c, logl, loga, logb, targets):
        """Compute evidential loss.

        All inputs are [N, 3] tensors covering valid pixels only.
        """
        a = self.active(loga) + 1.0 + 1e-6       # alpha
        b = self.active(logb) + 1e-6              # beta
        v = self.active(logl) + 1e-6              # nu / lambda

        if self.use_gs:
            nll = gs_l_nll(c, v, a, b, targets)
        else:
            nll = nig_nll(c, v, a, b, targets)

        reg = nig_reg(c, v, a, b, targets)
        return nll + self.lamb * reg


# ---------------------------------------------------------------------------
# Uncertainty helpers
# ---------------------------------------------------------------------------

def compute_std(logl, loga, logb):
    """Compute per-pixel prediction standard deviation.

    logl, loga, logb : Tensor of any shape
    Returns std of same shape as inputs.
    """
    a = torch.exp(loga) + 1.0 + 1e-6
    b = torch.exp(logb) + 1e-6
    v = torch.exp(logl) + 1e-6       # nu

    a_var = b / (a - 1)              # aleatoric uncertainty
    e_var = b / ((a - 1) * v)        # epistemic uncertainty
    return torch.sqrt(a_var + e_var)


# ==========================================================================
# Classification EDL — Dirichlet evidence (MSE + variance + KL)
# ==========================================================================

class EvidentialClassificationLoss(nn.Module):
    """EDL for soft-label classification (MSE-based NLL + variance + Dirichlet KL).

    Parameters
    ----------
    lamb : float
        KL regularisation weight.
    """

    def __init__(self, lamb=0.001):
        super().__init__()
        self.lamb = lamb

    def forward(self, logits, target_probs):
        """Compute EDL classification loss.

        logits       : [N, K] — per-pixel raw bin logits
        target_probs : [N, K] — Gaussian-smoothed bin distribution (sum=1)
        """
        alpha = F.softplus(logits) + 1.0                     # evidence → concentration
        S = alpha.sum(-1, keepdim=True)                      # total strength
        p = alpha / (S + 1e-8)                               # expected probabilities

        mse = ((target_probs - p) ** 2).sum(-1)              # prediction error
        var = (p * (1 - p) / (S + 1)).sum(-1)               # Dirichlet variance
        nll = mse + var                                       # expected MSE

        # KL with uniform Dirichlet Dir(1,…,1)
        alpha_unif = torch.ones_like(alpha)
        S_unif = alpha_unif.sum(-1)

        kl = (S.lgamma().squeeze(-1) - S_unif.lgamma()
              - (alpha.lgamma() - alpha_unif.lgamma()).sum(-1)
              + ((alpha - alpha_unif)
                 * (alpha.digamma() - S.digamma())).sum(-1))

        return (nll + self.lamb * kl).mean()


def compute_cls_uncertainty(logits):
    """Per-pixel per-coordinate uncertainty u = K / Σα.

    logits : [..., K]  bin logits
    Returns [..., 1]  uncertainty per pixel
    """
    alpha = F.softplus(logits) + 1.0
    return alpha.shape[-1] / (alpha.sum(-1) + 1e-8)


