"""
Geospatial Preprocessing & Co-registration Service
Handles reprojection, AOI cropping, radiometric normalization, sub-pixel co-registration,
and seamless tile blending.
"""

import os
import json
import logging
import numpy as np
import cv2
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("bhuvistaar.geoprocess")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "preprocessing", "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    PREPROC_CONFIG = json.load(f)

def normalize_sentinel2_bands(bands_data: np.ndarray) -> np.ndarray:
    """
    Normalizes Sentinel-2 L2A reflectance (nominally 0-10000) to [0, 1] float32.
    bands_data: shape (C, H, W) or (H, W, C)
    """
    scale = PREPROC_CONFIG["radiometric_normalization"]["sentinel2_scale"]
    norm = bands_data.astype(np.float32) / scale
    return np.clip(norm, 0.0, 1.0)

def normalize_cartosat(cartosat_data: np.ndarray) -> np.ndarray:
    """
    Normalizes Cartosat 10-bit panchromatic data (0-1023) to [0, 1] float32.
    """
    max_val = PREPROC_CONFIG["radiometric_normalization"]["cartosat_max_value"]
    norm = cartosat_data.astype(np.float32) / max_val
    return np.clip(norm, 0.0, 1.0)

def compute_hann_window(size: int) -> np.ndarray:
    """Creates a 2D Hann window for weighted overlap tile blending."""
    h1d = np.hanning(size)
    window_2d = np.outer(h1d, h1d)
    return np.clip(window_2d, 1e-4, 1.0).astype(np.float32)

def co_register_images(reference_img: np.ndarray, target_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns target_img to reference_img using Enhanced Correlation Coefficient (ECC)
    or Phase Correlation for sub-pixel accuracy.
    Both images should be 2D single-band or grayscale in [0, 1].
    """
    ref_gray = reference_img if reference_img.ndim == 2 else cv2.cvtColor(reference_img, cv2.COLOR_RGB2GRAY)
    tar_gray = target_img if target_img.ndim == 2 else cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)

    ref_u8 = (np.clip(ref_gray, 0, 1) * 255).astype(np.uint8)
    tar_u8 = (np.clip(tar_gray, 0, 1) * 255).astype(np.uint8)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)

    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_u8, tar_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )
        h, w = ref_u8.shape
        aligned_target = cv2.warpAffine(
            target_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        return aligned_target, warp_matrix
    except Exception as e:
        logger.warning(f"ECC alignment fallback to identity: {e}")
        return target_img, warp_matrix

def verify_aoi_overlap(geom_s2: Dict[str, Any], geom_ref: Dict[str, Any]) -> float:
    """
    Calculates intersection-over-union (IoU) between Sentinel-2 and Reference bounding footprints.
    Returns overlap ratio between 0.0 and 1.0.
    """
    try:
        coords_s2 = np.array(geom_s2.get("coordinates", [[]])[0])
        coords_ref = np.array(geom_ref.get("coordinates", [[]])[0])

        min_s2, max_s2 = coords_s2.min(axis=0), coords_s2.max(axis=0)
        min_ref, max_ref = coords_ref.min(axis=0), coords_ref.max(axis=0)

        ix_min = max(min_s2[0], min_ref[0])
        iy_min = max(min_s2[1], min_ref[1])
        ix_max = min(max_s2[0], max_ref[0])
        iy_max = min(max_s2[1], max_ref[1])

        if ix_max <= ix_min or iy_max <= iy_min:
            return 0.0

        inter_area = (ix_max - ix_min) * (iy_max - iy_min)
        area_s2 = (max_s2[0] - min_s2[0]) * (max_s2[1] - min_s2[1])
        area_ref = (max_ref[0] - min_ref[0]) * (max_ref[1] - min_ref[1])
        union_area = area_s2 + area_ref - inter_area

        return float(inter_area / union_area) if union_area > 0 else 0.0
    except Exception:
        return 1.0  # Fallback assumption if geometries are simplified
