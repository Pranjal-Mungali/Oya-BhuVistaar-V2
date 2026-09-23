"""
BhuVistaar - Enhanced Satellite Super-Resolution Training Engine.
4x Spatial Super-Resolution with L1 + VGG Perceptual Loss, Data Augmentation,
Cosine Annealing LR, and Validation PSNR/SSIM Tracking.
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2

from model import BhuVistaarNet
from utils import load_satellite_image


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
# 2. LOSS FUNCTIONS: L1 + VGG PERCEPTUAL + SOBEL EDGE
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


# ============================================================================
# 3. DATASET & AUGMENTATION PIPELINE
# ============================================================================

class SatelliteCropDataset(Dataset):
    """
    Generates paired HR and downsampled 4x LR patches from satellite images
    with geometric augmentations (horizontal flip, vertical flip, random 90-deg rotations).
    """
    def __init__(self, file_paths, crop_size=128, num_patches_per_image=50, scale=4, augment=True):
        self.patches_hr = []
        self.crop_size = crop_size
        self.scale = scale
        self.augment = augment

        for fp in file_paths:
            if not os.path.exists(fp):
                continue
            try:
                data, _, _ = load_satellite_image(fp)
                c, h, w = data.shape

                # Ensure image dimensions are at least crop_size
                if h < crop_size or w < crop_size:
                    new_h = max(h, crop_size)
                    new_w = max(w, crop_size)
                    resized = [cv2.resize(data[ci], (new_w, new_h), interpolation=cv2.INTER_CUBIC) for ci in range(c)]
                    data = np.stack(resized, axis=0)
                    h, w = new_h, new_w

                # Normalize to 3 RGB channels for training
                if c == 1:
                    data = np.repeat(data, 3, axis=0)
                elif c > 3:
                    data = data[:3]

                for _ in range(num_patches_per_image):
                    top = np.random.randint(0, h - crop_size + 1)
                    left = np.random.randint(0, w - crop_size + 1)
                    patch = data[:, top:top+crop_size, left:left+crop_size].copy()

                    # Data Augmentations
                    if self.augment:
                        # 1. Random Horizontal Flip
                        if np.random.rand() > 0.5:
                            patch = np.flip(patch, axis=2).copy()
                        # 2. Random Vertical Flip
                        if np.random.rand() > 0.5:
                            patch = np.flip(patch, axis=1).copy()
                        # 3. Random 90° Rotations (0, 90, 180, 270)
                        rot_k = np.random.choice([0, 1, 2, 3])
                        if rot_k > 0:
                            patch = np.rot90(patch, k=rot_k, axes=(1, 2)).copy()

                    self.patches_hr.append(patch)
            except Exception as e:
                print(f"Dataset crop warning for {fp}: {e}")

    def __len__(self):
        return len(self.patches_hr)

    def __getitem__(self, idx):
        hr = self.patches_hr[idx]
        c, h, w = hr.shape
        # Create downsampled 4x LR ground truth pair (128x128 -> 32x32)
        lr_h, lr_w = h // self.scale, w // self.scale
        lr = np.stack([cv2.resize(hr[ci], (lr_w, lr_h), interpolation=cv2.INTER_AREA) for ci in range(c)], axis=0)
        return torch.from_numpy(lr).float(), torch.from_numpy(hr).float()


# ============================================================================
# 4. TRAINING ENGINE
# ============================================================================

def train(
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 2e-4,
    scale: int = 4,
    crop_size: int = 128,
    patches_per_image: int = 60,
    vgg_weight: float = 0.1,
    val_split: float = 0.15,
    device_str: str = None
):
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================", flush=True)
    print(f"  BhuVistaar {scale}x Satellite Super-Resolution Training", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Compute Device   : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Scale Factor     : {scale}x (10m -> 2.5m GSD)", flush=True)
    print(f"Total Epochs     : {epochs}", flush=True)
    print(f"Batch Size       : {batch_size}", flush=True)
    print(f"Learning Rate    : {lr}", flush=True)
    print(f"Loss Combination : L1 Loss + {vgg_weight} * VGG Perceptual Loss + 0.1 * Edge Loss", flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, "samples")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_weights_path = os.path.join(models_dir, "bhuvistaar_sr_x4_best.pth")

    # Discover all satellite training scenes
    files = glob.glob(os.path.join(samples_dir, "*.*"))
    data_dir = os.path.join(base_dir, "data", "train")
    if os.path.exists(data_dir):
        files.extend(glob.glob(os.path.join(data_dir, "*.*")))
    temp_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "bhuvistaar_outputs")
    if os.path.exists(temp_dir):
        files.extend(glob.glob(os.path.join(temp_dir, "upload_*.*")))

    valid_files = [f for f in files if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
    print(f"Found {len(valid_files)} satellite training scenes.", flush=True)

    # Build dataset with augmentations
    full_dataset = SatelliteCropDataset(
        valid_files,
        crop_size=crop_size,
        num_patches_per_image=patches_per_image,
        scale=scale,
        augment=True
    )
    total_patches = len(full_dataset)
    val_size = max(4, int(total_patches * val_split))
    train_size = total_patches - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    lr_dim = crop_size // scale
    print(f"Total Patches    : {total_patches} (Train: {train_size}, Validation: {val_size})", flush=True)
    print(f"Resolution Pair  : LR ({lr_dim}x{lr_dim}) -> HR ({crop_size}x{crop_size})", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, train_size), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False, drop_last=False)

    # Initialize 4x BhuVistaar model
    model = BhuVistaarNet(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_blocks=6,
        upscale_factor=scale,
        pretrained=False
    ).to(device)

    # If existing 4x checkpoint exists, warm-start from it
    if os.path.exists(out_weights_path):
        try:
            ckpt = torch.load(out_weights_path, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"Warm-starting from existing weights: {out_weights_path}", flush=True)
        except Exception as e:
            print(f"Starting fresh weights: {e}", flush=True)

    # Loss Functions
    criterion_l1 = nn.L1Loss()
    criterion_vgg = VGGPerceptualLoss(device)
    criterion_edge = SobelEdgeLoss(device)

    # Optimizer & Cosine Annealing Learning Rate Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_psnr = 0.0
    best_val_ssim = 0.0
    best_val_loss = float("inf")

    print("\nStarting Training Loop...\n" + "-" * 75, flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        train_vgg = 0.0
        train_batches = 0

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()
            pred_hr = model(lr_batch)

            loss_l1 = criterion_l1(pred_hr, hr_batch)
            loss_edge = criterion_edge(pred_hr, hr_batch)

            if criterion_vgg.available:
                loss_vgg_val = criterion_vgg(pred_hr, hr_batch)
                loss = loss_l1 + vgg_weight * loss_vgg_val + 0.1 * loss_edge
                train_vgg += loss_vgg_val.item()
            else:
                loss = loss_l1 + 0.3 * loss_edge
                train_vgg += 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_l1 += loss_l1.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_train_l1 = train_l1 / max(train_batches, 1)
        avg_train_vgg = train_vgg / max(train_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        # Validation Phase: Track PSNR and SSIM
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_batches = 0

        with torch.no_grad():
            for val_lr, val_hr in val_loader:
                val_lr = val_lr.to(device)
                val_hr = val_hr.to(device)

                pred_val = model(val_lr)
                val_loss += criterion_l1(pred_val, val_hr).item()
                val_psnr += calculate_psnr(pred_val, val_hr)
                val_ssim += calculate_ssim(pred_val, val_hr)
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        avg_val_psnr = val_psnr / max(val_batches, 1)
        avg_val_ssim = val_ssim / max(val_batches, 1)

        # Logging
        is_best = avg_val_psnr > best_val_psnr
        best_marker = " [BEST]" if is_best else ""

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or is_best:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] "
                f"Train Loss: {avg_train_loss:.4f} (L1: {avg_train_l1:.4f}, VGG: {avg_train_vgg:.4f}) | "
                f"Val PSNR: {avg_val_psnr:.2f} dB | Val SSIM: {avg_val_ssim:.4f} | "
                f"LR: {current_lr:.2e}{best_marker}",
                flush=True
            )

        # Save Best Model Based on Validation PSNR
        if is_best or epoch == 1:
            best_val_psnr = avg_val_psnr
            best_val_ssim = avg_val_ssim
            best_val_loss = avg_val_loss

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_psnr": best_val_psnr,
                "val_ssim": best_val_ssim,
                "val_loss": best_val_loss,
                "scale": scale,
                "in_channels": 3,
                "out_channels": 3
            }, out_weights_path)

    print("-" * 75, flush=True)
    print(f"Training Complete.", flush=True)
    print(f"Best Validation PSNR : {best_val_psnr:.2f} dB", flush=True)
    print(f"Best Validation SSIM : {best_val_ssim:.4f}", flush=True)
    print(f"Saved Best Model to  : {out_weights_path}", flush=True)
    return out_weights_path


# ============================================================================
# 5. CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BhuVistaar 4x Satellite Super-Resolution Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (e.g., 5 for test, 100 for high quality)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (e.g. 16 on GPU, 4 on CPU)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate (default: 2e-4)")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution upscale factor (default: 4)")
    parser.add_argument("--crop_size", type=int, default=128, help="High-resolution patch size (default: 128)")
    parser.add_argument("--patches", type=int, default=60, help="Number of patches extracted per scene")
    parser.add_argument("--vgg_weight", type=float, default=0.1, help="Weight for VGG perceptual loss (default: 0.1)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        scale=args.scale,
        crop_size=args.crop_size,
        patches_per_image=args.patches,
        vgg_weight=args.vgg_weight,
        device_str=args.device
    )
