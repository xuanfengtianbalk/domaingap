"""L2SDG (Learning to Learn Single Domain Generalization) components.

Full WAE-style perturbation network following "Learning to Learn Single
Domain Generalization" (Qiao et al., CVPR 2020):
  - Encoder E: image -> latent style code z
  - Generator G: z -> residual perturbation delta (pixel space)
  - MMD penalty keeps z close to prior N(0, I)  (WAE)
  - L_norm keeps perturbation magnitude small

Perturbation is applied in pixel space: image denormalized -> + delta -> renorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _EncBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 4, 2, 1),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _DecBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 4, 2, 1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class WAE(nn.Module):
    """Encoder-decoder producing bounded image-space residual perturbation."""

    def __init__(self, latent_dim: int = 128, epsilon: float = 8.0, base_ch: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.epsilon = epsilon / 255.0
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        # encoder: 256 -> 4x4
        self.encoder = nn.Sequential(
            _EncBlock(3, base_ch),        # 128
            _EncBlock(base_ch, base_ch * 2),   # 64
            _EncBlock(base_ch * 2, base_ch * 4),  # 32
            _EncBlock(base_ch * 4, base_ch * 8),  # 16
            _EncBlock(base_ch * 8, base_ch * 8),  # 8
            _EncBlock(base_ch * 8, base_ch * 8),  # 4
        )
        self.fc_mu = nn.Linear(base_ch * 8 * 4 * 4, latent_dim)

        # generator: z -> residual (pixel space), tanh-bounded
        self.fc_z = nn.Linear(latent_dim, base_ch * 8 * 4 * 4)
        self.decoder = nn.Sequential(
            _DecBlock(base_ch * 8, base_ch * 8),   # 8
            _DecBlock(base_ch * 8, base_ch * 4),   # 16
            _DecBlock(base_ch * 4, base_ch * 2),   # 32
            _DecBlock(base_ch * 2, base_ch),       # 64
            _DecBlock(base_ch, base_ch),           # 128
            nn.ConvTranspose2d(base_ch, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x)
        z = self.fc_mu(h.flatten(1))
        return z

    def decode(self, z):
        h = self.fc_z(z).view(z.shape[0], -1, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        """x: normalized image [B,3,H,W]. Returns perturbed normalized image."""
        # denormalize to pixel space
        x_pix = x * self.std + self.mean
        z = self.encode(x)
        delta = self.decode(z) * self.epsilon
        x_pert_pix = torch.clamp(x_pix + delta, 0.0, 1.0)
        # renorm
        return (x_pert_pix - self.mean) / self.std

    def delta_norm(self, x):
        """||delta||^2 on the residual produced for input x."""
        z = self.encode(x)
        delta = self.decode(z) * self.epsilon
        return (delta ** 2).mean()


def mmd_rbf(z: torch.Tensor, z_prior: torch.Tensor, kernel_bandwidth: float = 1.0) -> torch.Tensor:
    """RBF-kernel MMD between latent z and prior samples."""
    z = z.contiguous()
    z_prior = z_prior.contiguous()

    def rbf(a, b):
        aa = (a * a).sum(dim=1, keepdim=True)
        bb = (b * b).sum(dim=1, keepdim=True)
        dist = aa - 2.0 * (a @ b.t()) + bb.t()
        return torch.exp(-dist / (2.0 * kernel_bandwidth ** 2))

    xx = rbf(z, z).mean()
    yy = rbf(z_prior, z_prior).mean()
    xy = rbf(z, z_prior).mean()
    return xx + yy - 2.0 * xy
