"""
BhuVistaar - Training Loss Functions & Metrics.
Includes L1 Loss, VGG-19 Perceptual Loss, Sobel Edge Gradient Loss, and GPU PSNR/SSIM.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. EVALUATION METRICS: PSNR & SSIM (GPU-Accelerated)
# ============================================================================

def create_gaussian_window(window_size: int = 11, sigma: float = 1.5, channel: int = 3) -> torch.Tensor:
    """Creates a 2D Gaussian filter kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """
    Computes Structural Similarity Index (SSIM) between img1 and img2 on PyTorch tensors.
    Expected input range: [0, 1], shape: (B, C, H, W).
    """
    channel = img1.size(1)
    window = create_gaussian_window(window_size, sigma, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return float(torch.clamp(ssim_map.mean(), 0.0, 1.0).item())


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)."""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10(1.0 / max(mse, 1e-10)))


# ============================================================================
# 2. LOSS FUNCTIONS: VGG PERCEPTUAL + SOBEL EDGE + COMPOSITE
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """
    VGG-19 based perceptual feature loss.
    Extracts deep perceptual representations from the relu3_3 layer (layer 16)
    to penalize structural and textural blurriness.
    """
    def __init__(self, device: torch.device):
        super(VGGPerceptualLoss, self).__init__()
        self.device = device
        self.available = False

        try:
            import torchvision.models as models
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:16].to(device)
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg
            self.available = True

            # Standard ImageNet normalization parameters for VGG
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
            print("Perceptual Loss: VGG-19 (relu3_3) initialized successfully on", device, flush=True)
        except Exception as e:
            print(f"Notice: VGG19 weights not yet cached ({e}). Falling back to Sobel Edge Loss.", flush=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.available:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Normalize to ImageNet distribution
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        feat_pred = self.vgg(pred_norm)
        feat_target = self.vgg(target_norm)
        return F.l1_loss(feat_pred, feat_target)


class SobelEdgeLoss(nn.Module):
    """Computes L1 difference between Sobel gradient magnitudes of prediction and target."""
    def __init__(self, device: torch.device):
        super(SobelEdgeLoss, self).__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.sobel_x = sobel_x.to(device)
        self.sobel_y = sobel_y.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_gray = torch.mean(pred, dim=1, keepdim=True)
        target_gray = torch.mean(target, dim=1, keepdim=True)

        gx_pred = F.conv2d(pred_gray, self.sobel_x, padding=1)
        gy_pred = F.conv2d(pred_gray, self.sobel_y, padding=1)
        grad_pred = torch.sqrt(gx_pred ** 2 + gy_pred ** 2 + 1e-6)

        gx_target = F.conv2d(target_gray, self.sobel_x, padding=1)
        gy_target = F.conv2d(target_gray, self.sobel_y, padding=1)
        grad_target = torch.sqrt(gx_target ** 2 + gy_target ** 2 + 1e-6)

        return F.l1_loss(grad_pred, grad_target)


class CompositeLoss(nn.Module):
    """
    Composite satellite super-resolution loss combining:
    L1 Pixel Loss + alpha * VGG Perceptual Loss + beta * Sobel Edge Loss.
    """
    def __init__(self, device: torch.device, vgg_weight: float = 0.1, edge_weight: float = 0.1):
        super(CompositeLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGPerceptualLoss(device)
        self.edge_loss = SobelEdgeLoss(device)
        self.vgg_weight = vgg_weight
        self.edge_weight = edge_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.l1_loss(pred, target)
        if self.vgg_weight > 0 and self.vgg_loss.available:
            loss = loss + self.vgg_weight * self.vgg_loss(pred, target)
        if self.edge_weight > 0:
            loss = loss + self.edge_weight * self.edge_loss(pred, target)
        return loss
