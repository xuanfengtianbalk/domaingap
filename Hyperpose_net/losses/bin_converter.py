"""
Regression-to-Classification Bin Converter (Torch)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinConverter(nn.Module):
    def __init__(self, sample_range, n_per_unit=30, max_sigma=0.25,
                 pad_factor=4.0, sigma_factor=1.5, min_bins=5):
        super().__init__()
        a, b = sample_range
        self.padded_range = (a - pad_factor * max_sigma, b + pad_factor * max_sigma)
        self.plen = self.padded_range[1] - self.padded_range[0]
        total_bins = max(int(n_per_unit * self.plen), min_bins)
        self.register_buffer('bin_centers', torch.linspace(self.padded_range[0], self.padded_range[1], total_bins))
        self.bin_width = self.plen / (total_bins - 1) if total_bins > 1 else self.plen
        self.sigma = sigma_factor * self.bin_width

    @property
    def total_bins(self):
        return len(self.bin_centers)

    def _gaussian_pdf(self, x, mu, sigma):
        coeff = 1.0 / (sigma * (2 * torch.pi) ** 0.5)
        return coeff * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def value_to_bins(self, value):
        """value: (N,) or scalar → (N, total_bins) or (total_bins,)"""
        value = torch.atleast_1d(value)
        pdf = self._gaussian_pdf(self.bin_centers, value.unsqueeze(-1), self.sigma)
        pdf_sum = pdf.sum(dim=-1, keepdim=True) + 1e-30
        probs = pdf / pdf_sum
        if probs.shape[0] == 1:
            return probs.squeeze(0)
        return probs

    def bins_to_value(self, probs, threshold=1e-4, fg_threshold=0.0):
        """probs: (..., total_bins) → (... ,)"""
        probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)
        denom = probs.sum(dim=-1, keepdim=True) + 1e-12
        probs = probs / denom
        value = (probs * self.bin_centers).sum(dim=-1)
        max_prob = probs.max(dim=-1).values
        value = torch.where(max_prob < fg_threshold, torch.tensor(float('nan'), device=value.device), value)
        return value

    def loss_js(self, pred_logits, gt_value):
        gt_probs = self.value_to_bins(gt_value)
        T = gt_probs.shape[-1]
        gt = torch.clamp(gt_probs.view(-1, T), 1e-12, 1.0)
        pred = torch.clamp(F.softmax(pred_logits.view(-1, T), dim=-1), 1e-12, 1.0)
        m = 0.5 * (gt + pred)
        kl1 = F.kl_div(torch.log(m), gt, reduction='batchmean')
        kl2 = F.kl_div(torch.log(m), pred, reduction='batchmean')
        return 0.5 * (kl1 + kl2)

    def loss_ce(self, pred_logits, gt_value):
        gt_probs = self.value_to_bins(gt_value)
        T = gt_probs.shape[-1]
        gt = torch.clamp(gt_probs.view(-1, T), 1e-12, 1.0)
        return F.kl_div(F.log_softmax(pred_logits.view(-1, T), dim=-1), gt, reduction='batchmean')
