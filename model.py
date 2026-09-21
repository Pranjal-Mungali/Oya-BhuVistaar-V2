"""
BhuVistaar - Satellite Imagery Super-Resolution Model
Team: The Outliers (Smart India Hackathon 2026)

Architecture:
- Deep Residual CNN with Sub-Pixel Convolution (PixelShuffle 2x)
- Dynamic multi-band support: RGB (3-band) and RGB+NIR (4-band), plus Panchromatic (1-band)
- Integrated Spatial Dropout for Epistemic Uncertainty Estimation
- Monte Carlo Dropout inference engine (15 stochastic forward passes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class ResidualBlock(nn.Module):
    """
    Residual Block with dual Conv2D, PReLU activation, and Spatial Dropout
    for robust feature extraction and uncertainty sampling.
    """
    def __init__(self, channels: int = 64, dropout_rate: float = 0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return residual + out * 0.2  # Residual scaling for numerical stability


class BhuVistaarNet(nn.Module):
    """
    2x Super-Resolution Network for Multi-Band Earth Observation Imagery.
    Features:
    - Dynamic channel adaptation (RGB, RGB+NIR, Panchromatic)
    - Residual feature backbone with MC Dropout
    - Sub-pixel Convolution (PixelShuffle) 2x upscaler
    - Global residual skip connection with bicubic reference
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 6,
        upscale_factor: int = 2,
        dropout_rate: float = 0.2
    ):
        super(BhuVistaarNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor

        # Head: Feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # Body: Deep Residual Blocks with MC Dropout
        self.body = nn.ModuleList([
            ResidualBlock(channels=num_features, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        self.body_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Upsampler: Sub-pixel convolution (PixelShuffle)
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU()
        )

        # Tail: High-frequency reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(num_features // 2, out_channels, kernel_size=3, padding=1)
        )

        self._initialize_weights()
        self._load_pretrained_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Tail layer init for smooth residual edge enhancement
        nn.init.normal_(self.tail[-1].weight, std=0.01)
        if self.tail[-1].bias is not None:
            nn.init.zeros_(self.tail[-1].bias)

    def _load_pretrained_weights(self):
        """Loads trained weights from weights/bhuvistaar_sr_2x.pth if available."""
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, "weights", "bhuvistaar_sr_2x.pth")
        if not os.path.exists(weights_path):
            return

        try:
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model_dict = self.state_dict()

            # Adapt weights if channel counts differ
            adapted_dict = {}
            for k, v in state_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        adapted_dict[k] = v
                    elif "head.0.weight" in k:
                        # Head input channel adaptation
                        cur_in = model_dict[k].shape[1]
                        v_in = v.shape[1]
                        if cur_in == 1 and v_in == 3:
                            adapted_dict[k] = v.mean(dim=1, keepdim=True)
                        elif cur_in == 4 and v_in == 3:
                            adapted_dict[k] = torch.cat([v, v[:, :1, :, :]], dim=1)
                    elif "tail.2.weight" in k:
                        # Tail output channel adaptation
                        cur_out = model_dict[k].shape[0]
                        v_out = v.shape[0]
                        if cur_out == 1 and v_out == 3:
                            adapted_dict[k] = v.mean(dim=0, keepdim=True)
                        elif cur_out == 4 and v_out == 3:
                            adapted_dict[k] = torch.cat([v, v[:1, :, :, :]], dim=0)
                    elif "tail.2.bias" in k and model_dict[k].shape != v.shape:
                        cur_out = model_dict[k].shape[0]
                        if cur_out == 1:
                            adapted_dict[k] = v.mean(dim=0, keepdim=True)
                        elif cur_out == 4:
                            adapted_dict[k] = torch.cat([v, v[:1]], dim=0)

            model_dict.update(adapted_dict)
            self.load_state_dict(model_dict)
            print(f"[BhuVistaarNet] Loaded pretrained satellite SR weights for {self.in_channels}-channel model.")
        except Exception as e:
            print(f"[BhuVistaarNet] Could not load weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with global residual connection.
        x: [B, C, H, W] in range [0, 1]
        returns: [B, C, 2*H, 2*W] in range [0, 1]
        """
        # Global bicubic baseline
        bicubic_baseline = F.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False
        )

        h = self.head(x)
        res = h
        for block in self.body:
            res = block(res)
        res = self.body_conv(res) + h

        up = self.upsampler(res)
        residual_detail = self.tail(up)

        # Super-resolved output = baseline + learned high-frequency residual detail pop
        out = bicubic_baseline + residual_detail * 2.2
        return torch.clamp(out, 0.0, 1.0)


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enables Dropout layers while keeping Normalization layers in eval mode.
    Crucial for valid Monte Carlo Dropout Bayesian approximation.
    """
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
    Performs Monte Carlo Dropout sampling across multiple stochastic passes.

    Args:
        model: BhuVistaarNet instance.
        input_tensor: [B, C, H, W] FloatTensor scaled to [0, 1].
        num_passes: Number of forward passes (default: 15).
        device: Torch device to execute on.

    Returns:
        mean_prediction: [B, C, 2H, 2W] Expected Super-Resolved image.
        uncertainty_map: [B, 1, 2H, 2W] Epistemic uncertainty (std deviation).
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    input_tensor = input_tensor.to(device)

    # Enable Monte Carlo Dropout
    enable_mc_dropout(model)

    predictions = []
    with torch.no_grad():
        for _ in range(num_passes):
            pred = model(input_tensor)
            predictions.append(pred)

    # Stack: [num_passes, B, C, 2H, 2W]
    stacked = torch.stack(predictions, dim=0)

    # Predictive mean (Super-resolved image)
    mean_sr = torch.mean(stacked, dim=0)

    # Predictive variance & standard deviation (Epistemic uncertainty)
    channel_variance = torch.var(stacked, dim=0)  # [B, C, 2H, 2W]
    uncertainty_map = torch.mean(channel_variance, dim=1, keepdim=True).sqrt()  # [B, 1, 2H, 2W]

    # Reset model to standard eval mode
    model.eval()

    return mean_sr, uncertainty_map


class BhuVistaarModelManager:
    """
    Manages model instances for different band configurations (3-band RGB vs 4-band RGB+NIR)
    with caching and device management.
    """
    def __init__(self, device: torch.device = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self._models: Dict[int, BhuVistaarNet] = {}

    def get_model(self, channels: int) -> BhuVistaarNet:
        if channels not in self._models:
            model = BhuVistaarNet(in_channels=channels, out_channels=channels)
            model.to(self.device)
            model.eval()
            self._models[channels] = model
        return self._models[channels]

    def super_resolve(
        self,
        input_tensor: torch.Tensor,
        num_passes: int = 15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = input_tensor.shape[1]
        model = self.get_model(channels)
        return mc_dropout_inference(model, input_tensor, num_passes=num_passes, device=self.device)

    def warmup(self):
        """Warms up CUDA context and caches models for 1, 3, and 4 channel inputs."""
        print(f"[BhuVistaar] Warming up compute pipeline on device: {self.device}...")
        for ch in [1, 3, 4]:
            m = self.get_model(ch)
            dummy = torch.zeros((1, ch, 32, 32), device=self.device)
            with torch.no_grad():
                _ = m(dummy)
        print("[BhuVistaar] Warmup complete. GPU memory and kernels initialized.")
