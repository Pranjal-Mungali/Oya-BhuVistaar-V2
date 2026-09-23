"""
Dataset Preparation for 4-band Dual-Pathway Super-Resolution (4x)
Generates paired LR (32x32) and HR (128x128) patches from GeoTIFF scenes.
"""

import os
import glob
import logging
import numpy as np
import rasterio
import cv2

logger = logging.getLogger("bhuvistaar.prepare_dataset")

def extract_patches(
    input_tif_path: str,
    output_dir: str,
    patch_size_hr: int = 128,
    scale_factor: int = 4,
    stride: int = 64
):
    """
    Extracts paired LR and HR patches from a multispectral GeoTIFF.
    """
    os.makedirs(os.path.join(output_dir, "hr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lr"), exist_ok=True)

    patch_size_lr = patch_size_hr // scale_factor

    with rasterio.open(input_tif_path) as src:
        data = src.read()  # Shape: (C, H, W)
        c, h, w = data.shape
        if c < 4:
            # Expand to 4 bands if only 3 bands present
            extra = np.mean(data[:3], axis=0, keepdims=True)
            data = np.concatenate([data, extra], axis=0)
            c = 4

        count = 0
        for y in range(0, h - patch_size_hr + 1, stride):
            for x in range(0, w - patch_size_hr + 1, stride):
                hr_patch = data[:4, y:y + patch_size_hr, x:x + patch_size_hr]
                
                # Downsample 4x to create LR
                lr_bands = []
                for b in range(4):
                    lr_b = cv2.resize(
                        hr_patch[b].astype(np.float32),
                        (patch_size_lr, patch_size_lr),
                        interpolation=cv2.INTER_AREA
                    )
                    lr_bands.append(lr_b)
                lr_patch = np.stack(lr_bands, axis=0)

                base_name = f"{os.path.splitext(os.path.basename(input_tif_path))[0]}_p{count:04d}"
                np.save(os.path.join(output_dir, "hr", f"{base_name}_hr.npy"), hr_patch.astype(np.float32))
                np.save(os.path.join(output_dir, "lr", f"{base_name}_lr.npy"), lr_patch.astype(np.float32))
                count += 1

    logger.info(f"Generated {count} paired patches from {input_tif_path}")
    return count

def prepare_all_scenes(data_dir: str, output_dir: str):
    tifs = glob.glob(os.path.join(data_dir, "*.tif"))
    total = 0
    for tif in tifs:
        total += extract_patches(tif, output_dir)
    print(f"Total patches generated: {total}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, "data", "train")
    out_dir = os.path.join(base_dir, "data", "patches")
    prepare_all_scenes(data_dir, out_dir)
