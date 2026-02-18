"""
Custom loss functions for spatio-temporal forecasting.
Combines pixel-wise MSE with structural similarity (SSIM) to preserve edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(size=11, sigma=1.5, channels=1):
    """Create a 2D Gaussian kernel for SSIM."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    g = g / g.sum()
    return g.view(1, 1, size, size).repeat(channels, 1, 1, 1)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss: returns 1 - SSIM (so lower is better)."""
    def __init__(self, window_size=11, channels=4):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer('window', _gaussian_kernel(window_size, 1.5, channels))

    def forward(self, pred, target):
        # Handle multi-step predictions: [B, T, C, H, W] -> flatten to [B*T, C, H, W]
        if pred.dim() == 5:
            B, T, C, H, W = pred.shape
            pred = pred.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)

        C = pred.size(1)
        
        # Ensure window matches channels
        if self.window.size(0) != C:
            self.window = _gaussian_kernel(self.window_size, 1.5, C).to(pred.device)

        mu1 = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=C)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=C) - mu12

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """MSE + α * SSIM Loss."""
    def __init__(self, ssim_weight=0.3, channels=4):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(channels=channels)
        self.alpha = ssim_weight

    def forward(self, pred, target):
        mse_val = self.mse(pred, target)
        ssim_val = self.ssim(pred, target)
        return mse_val + self.alpha * ssim_val
