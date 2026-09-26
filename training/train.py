"""
BhuVistaar - Satellite Super-Resolution Training Engine.
Executes 4x spatial super-resolution training with L1, VGG-19 perceptual, and Sobel edge losses.
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Ensure parent directory is in sys.path for modular imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from training.losses import (
    calculate_ssim,
    calculate_psnr,
    VGGPerceptualLoss,
    SobelEdgeLoss
)
from training.dataset import SatelliteCropDataset

try:
    from app.model import BhuVistaarNet
except ImportError:
    from model import BhuVistaarNet


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

    samples_dir = os.path.join(BASE_DIR, "samples")
    weights_dir = os.path.join(BASE_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    out_weights_path = os.path.join(weights_dir, "bhuvistaar_sr_x4_best.pth")

    # Discover satellite training scenes
    files = glob.glob(os.path.join(samples_dir, "*.*"))
    data_dir = os.path.join(BASE_DIR, "data", "train")
    if os.path.exists(data_dir):
        files.extend(glob.glob(os.path.join(data_dir, "*.*")))

    valid_files = [f for f in files if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
    print(f"Found {len(valid_files)} satellite training scenes.", flush=True)

    if len(valid_files) == 0:
        raise RuntimeError("No training images found in samples/ or data/train/.")

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

    # Initialize model
    model = BhuVistaarNet(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_blocks=6,
        upscale_factor=scale,
        pretrained=False
    ).to(device)

    # Warm-start if checkpoint exists
    if os.path.exists(out_weights_path):
        try:
            ckpt = torch.load(out_weights_path, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"Warm-starting from existing weights: {out_weights_path}", flush=True)
        except Exception as e:
            print(f"Starting fresh weights: {e}", flush=True)

    criterion_l1 = nn.L1Loss()
    criterion_vgg = VGGPerceptualLoss(device)
    criterion_edge = SobelEdgeLoss(device)

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

        # Validation Phase
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BhuVistaar 4x Satellite Super-Resolution Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution upscale factor")
    parser.add_argument("--crop_size", type=int, default=128, help="High-resolution patch size")
    parser.add_argument("--patches", type=int, default=60, help="Number of patches extracted per scene")
    parser.add_argument("--vgg_weight", type=float, default=0.1, help="Weight for VGG perceptual loss")
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
