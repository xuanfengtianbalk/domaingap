"""
Regression-to-Classification Bin Converter (Torch)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinConverter(nn.Module):
    def __init__(self, sample_range, n_per_unit=30, max_sigma=0.25,
                 pad_factor=4.0, sigma_factor=1.5, min_bins=5, use_mask=True,
                 loss_reduction='mean'):
        super().__init__()
        self.use_mask = use_mask
        self.loss_reduction = loss_reduction
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

    def mask_from_probs(self, probs, threshold=0.5):
        max_per_channel = probs.max(dim=-1).values  # (N, 3)
        return (max_per_channel > threshold).all(dim=-1)  # (N,)

    def compute_sigma(self, probs):
        """
        probs: (H, W, C) 概率数组，每个通道和为 1
        返回: sigma (H, W)
        """
        C = probs.shape[-1]
        # 类别索引作为取值
        indices = np.arange(C)  # shape (C,)
        # 期望 mu: (H, W)
        mu = np.sum(probs * indices, axis=-1)
        # 方差: sum(p * (i - mu)^2)
        variance = np.sum(probs * (indices - mu[..., None]) ** 2, axis=-1)
        sigma = np.sqrt(variance)
        return sigma
    def bins_to_value(self, probs, threshold=1e-4, fg_threshold= 10):
        probs = np.asarray(probs.detach().cpu())

        axis = -1

        denom = probs.sum(axis=axis, keepdims=True)
        denom = np.where(denom < threshold, 1.0, denom)
        probs = probs / denom
        value = np.sum(probs * self.bin_centers.cpu().numpy(), axis=axis)

        background_val = float('nan')

        max_per_channel = probs.max(axis=-1)



        sigma=self.compute_sigma(probs)
        print(sigma)
        value = np.where((sigma > fg_threshold), background_val, value)
        return value

    def loss_js(self, pred_logits, gt_value):
        gt_probs = self.value_to_bins(gt_value)
        T = gt_probs.shape[-1]
        gt = torch.clamp(gt_probs.view(-1, T), 1e-12, 1.0)
        pred = torch.clamp(F.softmax(pred_logits.view(-1, T), dim=-1), 1e-12, 1.0)
        m = 0.5 * (gt + pred)
        kl1 = F.kl_div(torch.log(m), gt, reduction='none').sum(dim=-1)  # per-pixel sum
        kl2 = F.kl_div(torch.log(m), pred, reduction='none').sum(dim=-1)
        js_per_pixel = 0.5 * (kl1 + kl2)  # (N*3,)
        if self.loss_reduction == 'mean':
            return js_per_pixel.mean()
        else:  # 'sum': per-pixel sum then avg (like dsntnn)
            return js_per_pixel.sum() / gt_probs.shape[0]  # divide by N

    def loss_ce(self, pred_logits, gt_value):
        gt_probs = self.value_to_bins(gt_value)
        T = gt_probs.shape[-1]
        gt = torch.clamp(gt_probs.view(-1, T), 1e-12, 1.0)
        ce_per_pixel = F.kl_div(F.log_softmax(pred_logits.view(-1, T), dim=-1), gt, reduction='none').sum(dim=-1)
        if self.loss_reduction == 'mean':
            return ce_per_pixel.mean()
        else:
            return ce_per_pixel.sum() / gt_probs.shape[0]  # divide by N
