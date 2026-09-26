"""
BhuVistaar - Satellite Dataset & Augmentation Pipeline.
Generates paired HR and downsampled LR patches from multispectral & RGB satellite imagery.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from app.utils import load_satellite_image
except ImportError:
    from utils import load_satellite_image


class SatelliteCropDataset(Dataset):
    """
    Generates paired HR and downsampled 4x LR patches from satellite images
    with geometric augmentations (horizontal flip, vertical flip, random 90-deg rotations).
    """
    def __init__(
        self,
        file_paths,
        crop_size: int = 128,
        num_patches_per_image: int = 50,
        scale: int = 4,
        augment: bool = True
    ):
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

                # Normalize to 3 RGB channels for backbone training
                if c == 1:
                    data = np.repeat(data, 3, axis=0)
                elif c > 3:
                    data = data[:3]

                for _ in range(num_patches_per_image):
                    top = np.random.randint(0, h - crop_size + 1)
                    left = np.random.randint(0, w - crop_size + 1)
                    patch = data[:, top:top + crop_size, left:left + crop_size].copy()

                    # Data Augmentations
                    if self.augment:
                        # 1. Random Horizontal Flip
                        if np.random.rand() > 0.5:
                            patch = np.flip(patch, axis=2).copy()
                        # 2. Random Vertical Flip
                        if np.random.rand() > 0.5:
                            patch = np.flip(patch, axis=1).copy()
                        # 3. Random 90-degree Rotations (0, 90, 180, 270)
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
        # Create downsampled 4x LR ground truth pair (e.g. 128x128 -> 32x32)
        lr_h, lr_w = h // self.scale, w // self.scale
        lr = np.stack([cv2.resize(hr[ci], (lr_w, lr_h), interpolation=cv2.INTER_AREA) for ci in range(c)], axis=0)
        return torch.from_numpy(lr).float(), torch.from_numpy(hr).float()
