import math
import torch
import torch.nn as nn

class SmoothDepthLoss(nn.Module):

    def __init__(self, alpha: float=10.0, eps: float=0.001, reduction: str='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor | None=None):
        dx = depth.diff(dim=-1)
        dy = depth.diff(dim=-2)
        rgb_dx = torch.mean(torch.abs(rgb.diff(dim=-1)), dim=1, keepdim=True)
        rgb_dy = torch.mean(torch.abs(rgb.diff(dim=-2)), dim=1, keepdim=True)
        wx = torch.exp(-self.alpha * rgb_dx)
        wy = torch.exp(-self.alpha * rgb_dy)

        def charbonnier(g):
            return torch.sqrt(g * g + self.eps * self.eps)
        loss_x = wx * charbonnier(dx)
        loss_y = wy * charbonnier(dy)
        if mask is not None:
            mx = (mask[..., :, :, :-1] * mask[..., :, :, 1:]).clamp(0, 1)
            my = (mask[..., :, :-1, :] * mask[..., :, 1:, :]).clamp(0, 1)
            loss_x = loss_x * mx
            loss_y = loss_y * my
            denom = mx.sum() + my.sum()
        else:
            denom = loss_x.numel() + loss_y.numel()
        total = loss_x.sum() + loss_y.sum()
        if self.reduction == 'sum':
            return total
        return total / (denom + 1e-08)
