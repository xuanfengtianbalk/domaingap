"""BYOL contrastive head for domain generalization. Operates on decoder features."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Projector: decoder feature -> embedding space"""
    def __init__(self, in_dim=256, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """Predictor: embedding -> embedding. Asymmetric (2 layers) for BYOL"""
    def __init__(self, dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOLHead(nn.Module):
    """BYOL contrastive head. input: decoder feature [B, C, H, W]"""
    def __init__(self, in_channels=256, proj_dim=128, pred_hidden=64):
        super().__init__()
        self.projector = Projector(in_channels, proj_dim)
        self.predictor = Predictor(proj_dim, pred_hidden)

    def forward(self, feat1, feat2):
        """
        feat1, feat2: [B, C, H, W] from decoder (same image, different augmentations)
        Returns: BYOL loss (symmetric cosine similarity loss)
        """
        z1 = self.projector(feat1.mean(dim=[-2, -1]))
        z2 = self.projector(feat2.mean(dim=[-2, -1]))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = self._byol_loss(p1, z2.detach()) + self._byol_loss(p2, z1.detach())
        return loss * 0.5

    def _byol_loss(self, p, z):
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()
