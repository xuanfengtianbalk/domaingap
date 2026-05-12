import torch.nn as nn
import torch
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=3, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target):
        """
        output: [B, 224, 224], raw logits (no sigmoid)
        target: [B, 224, 224], 0 or 1
        """
        prob = torch.sigmoid(output)
        target = target.float()

        pt = prob * target + (1 - prob) * (1 - target)
        w = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = -w * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CoorsLoss(nn.Module):
    """Coordinate loss with normalization to [-1, 1]"""
    def __init__(self, scale_x=0.575, scale_y=0.565, scale_z=0.225):
        super().__init__()
        self.register_buffer('scale', torch.tensor([scale_x, scale_y, scale_z]))
        self.criterion = torch.nn.L1Loss()

    def forward(self, output, target):
        """Normalize target before computing loss"""
        target = target / self.scale.to(target.device)
        return self.criterion(output, target)

    @torch.no_grad()
    def post_process(self, output):
        """Denormalize output back to real-world coordinates"""
        return output * self.scale.to(output.device)