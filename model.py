"""
BhuVistaar - High-Fidelity 4x Super-Resolution & Monte Carlo Dropout Engine.
Combines deep 23-block RRDBNet (Real-ESRGAN architecture) with spatial Monte Carlo Dropout
to deliver razor-sharp spatial enhancement alongside Bayesian epistemic uncertainty mapping.
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "RealESRGAN_x4plus.pth")


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with integrated spatial dropout for Monte Carlo sampling."""
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, dropout_p: float = 0.06):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(self.dropout(torch.cat((x, x1, x2, x3, x4), 1)))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, dropout_p: float = 0.06):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, dropout_p)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, dropout_p)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RealESRGAN_MC(nn.Module):
    """
    Real-ESRGAN Generator with integrated Monte Carlo Dropout.
    23 deep RRDB blocks producing ultra-sharp satellite imagery reconstruction.
    """
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        dropout_p: float = 0.06
    ):
        super(RealESRGAN_MC, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch, dropout_p) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 4x upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# Backward compatible alias
BhuVistaarNet = RealESRGAN_MC


def enable_mc_dropout(model: nn.Module) -> None:
    """Activates dropout during inference while keeping batchnorm in eval mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def mc_dropout_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_passes: int = 15,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs deterministic inference for the razor-sharp super-resolved output,
    and Monte Carlo Dropout sampling across stochastic passes for epistemic uncertainty.
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    input_tensor = input_tensor.to(device)

    # 1. Deterministic high-fidelity base output
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.eval()

    with torch.no_grad():
        base_sr = model(input_tensor).clamp(0.0, 1.0)

    # 2. Monte Carlo Dropout passes for Epistemic Uncertainty Mapping
    enable_mc_dropout(model)

    predictions = []
    with torch.no_grad():
        for _ in range(num_passes):
            pred = model(input_tensor).clamp(0.0, 1.0)
            predictions.append(pred)

    stacked = torch.stack(predictions, dim=0)
    channel_variance = torch.var(stacked, dim=0)
    uncertainty_map = torch.mean(channel_variance, dim=1, keepdim=True).sqrt()

    # Blend deterministic base with ensemble mean for peak sharpness & consistency
    mean_sr = 0.7 * base_sr + 0.3 * torch.mean(stacked, dim=0)
    mean_sr = torch.clamp(mean_sr, 0.0, 1.0)

    model.eval()
    return mean_sr, uncertainty_map


class BhuVistaarModelManager:
    """Manages Real-ESRGAN instance, multi-channel handling (RGB & NIR), and device execution."""
    def __init__(self, device: torch.device = None, weights_path: str = WEIGHTS_PATH):
        if device is None:
            if torch.cuda.is_available():
                try:
                    torch.cuda.init()
                    _ = torch.zeros(1, device='cuda')
                    device = torch.device('cuda')
                except Exception as e:
                    print(f"[ModelManager] CUDA init notice ({e}), using CPU.")
                    device = torch.device('cpu')
            else:
                device = torch.device('cpu')
        self.device = device
        self.weights_path = weights_path
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            self._model = RealESRGAN_MC(num_in_ch=3, num_out_ch=3, scale=4).to(self.device)
            if os.path.exists(self.weights_path):
                ckpt = torch.load(self.weights_path, map_location=self.device)
                state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
                self._model.load_state_dict(state_dict, strict=True)
                print(f"[ModelManager] Loaded RealESRGAN weights with 100% strict match on {self.device}")
            else:
                print(f"[ModelManager] Warning: Weights file {self.weights_path} not found.")
            self._model.eval()
        except Exception as e:
            print(f"[ModelManager] Model load error: {e}")

    def super_resolve(
        self,
        input_tensor: torch.Tensor,
        num_passes: int = 15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Super-resolves input tensor [1, C, H, W] to 4x resolution.
        Handles 1-band (mono), 3-band (RGB), and 4-band (RGB + NIR) seamlessly.
        """
        b, c, h, w = input_tensor.shape

        if c == 1:
            # Replicate grayscale to 3 channels, super resolve, take mean
            rgb_in = input_tensor.repeat(1, 3, 1, 1)
            sr_3ch, unc = mc_dropout_inference(self._model, rgb_in, num_passes=num_passes, device=self.device)
            return sr_3ch[:, :1, :, :], unc
        elif c == 3:
            return mc_dropout_inference(self._model, input_tensor, num_passes=num_passes, device=self.device)
        elif c >= 4:
            # 4-band: super-resolve RGB channels through RealESRGAN
            rgb_in = input_tensor[:, :3, :, :]
            sr_rgb, unc = mc_dropout_inference(self._model, rgb_in, num_passes=num_passes, device=self.device)

            # Super-resolve NIR band with matched structural detail
            nir_in = input_tensor[:, 3:4, :, :].repeat(1, 3, 1, 1)
            with torch.no_grad():
                self._model.eval()
                sr_nir_3ch = self._model(nir_in.to(self.device)).clamp(0.0, 1.0)
            sr_nir = sr_nir_3ch[:, :1, :, :]

            sr_4band = torch.cat([sr_rgb, sr_nir], dim=1)
            return sr_4band, unc
        else:
            return mc_dropout_inference(self._model, input_tensor, num_passes=num_passes, device=self.device)

    def warmup(self):
        """Initializes GPU memory and warm caches."""
        try:
            dummy = torch.zeros((1, 3, 32, 32), device=self.device)
            with torch.no_grad():
                _ = self._model(dummy)
        except Exception as e:
            print(f"[ModelManager] Warmup note: {e}")
