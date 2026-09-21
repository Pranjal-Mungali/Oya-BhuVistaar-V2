"""
BhuVistaar - Geospatial Processing & Satellite Imagery Utilities

Features:
- Multi-band satellite image I/O via rasterio and OpenCV/PIL
- Support for .tif, .tiff, .png, .jpg, .jpeg
- Robust bit-depth normalization (uint8, uint16, int16, float32)
- Percentile-based dynamic range adjustment for Sentinel-2 / Landsat
- Model tensor conversion and viewable RGB / False-Color NIR rendering
- Publication-quality uncertainty heatmap generation with Matplotlib
- Scientific metrics: PSNR, SSIM, and Laplacian Sharpness Variance
- GeoTIFF exporter preserving georeferencing metadata
"""

import os
import io
import tempfile
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def normalize_band(
    band: np.ndarray,
    percentile_clip: bool = True,
    p_low: float = 1.0,
    p_high: float = 99.0
) -> np.ndarray:
    """
    Normalizes a single satellite band to [0.0, 1.0] with adaptive contrast stretch.
    Handles uint8, uint16 (e.g. Sentinel-2 / Landsat L2A), int16, and float32.
    """
    band = band.astype(np.float32)
    # Remove NaNs or Infs if present
    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)

    if percentile_clip and band.max() > band.min():
        low_val = np.percentile(band, p_low)
        high_val = np.percentile(band, p_high)
        if high_val > low_val:
            band = np.clip(band, low_val, high_val)
            band = (band - low_val) / (high_val - low_val)
            return band

    # Fallback to direct min-max normalization
    b_min, b_max = band.min(), band.max()
    if b_max > b_min:
        return (band - b_min) / (b_max - b_min)
    return np.zeros_like(band, dtype=np.float32)


def load_satellite_image(
    file_path: str
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Loads multi-band satellite images or standard photography.
    Supports GeoTIFF (.tif, .tiff), PNG, JPG, JPEG.

    Returns:
        normalized_data: [C, H, W] float32 array in [0.0, 1.0].
        metadata: Dict containing georeferencing, band count, original shape, bit depth.
        preview_rgb: [H, W, 3] uint8 RGB image for immediate UI display.
    """
    ext = os.path.splitext(file_path)[1].lower()
    metadata: Dict[str, Any] = {
        "file_name": os.path.basename(file_path),
        "extension": ext,
        "is_geotiff": False,
        "crs": None,
        "transform": None,
        "dtype": None,
        "nodata": None
    }

    raw_bands = []

    # 1. Attempt GeoTIFF loading with Rasterio if available and appropriate
    if HAS_RASTERIO and ext in [".tif", ".tiff"]:
        try:
            with rasterio.open(file_path) as src:
                metadata["is_geotiff"] = True
                metadata["crs"] = str(src.crs) if src.crs else "Non-projected / Local"
                metadata["transform"] = src.transform
                metadata["dtype"] = str(src.dtypes[0])
                metadata["nodata"] = src.nodata
                metadata["width"] = src.width
                metadata["height"] = src.height
                metadata["count"] = src.count

                # Read all bands: [Count, H, W]
                data = src.read()
                raw_bands = [data[i] for i in range(src.count)]
        except Exception as e:
            # Fall back to OpenCV / PIL if rasterio encounters a non-standard TIFF
            pass

    # 2. Fallback to OpenCV / PIL for standard images or failed TIFFs
    if len(raw_bands) == 0:
        # Try OpenCV with UNCHANGED flag to preserve 16-bit or alpha channels
        cv_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if cv_img is not None:
            metadata["dtype"] = str(cv_img.dtype)
            if cv_img.ndim == 2:
                # Grayscale / 1-band
                raw_bands = [cv_img]
            elif cv_img.ndim == 3:
                # In OpenCV, multi-channel is BGR or BGRA
                channels = cv_img.shape[2]
                if channels == 3:
                    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    raw_bands = [rgb[:, :, i] for i in range(3)]
                elif channels >= 4:
                    # Treat first 3 as BGR->RGB, 4th as NIR / Alpha
                    b, g, r = cv_img[:, :, 0], cv_img[:, :, 1], cv_img[:, :, 2]
                    nir = cv_img[:, :, 3]
                    raw_bands = [r, g, b, nir]
            metadata["count"] = len(raw_bands)
            metadata["height"] = cv_img.shape[0]
            metadata["width"] = cv_img.shape[1]
        else:
            # Final fallback: Pillow
            pil_img = Image.open(file_path)
            metadata["dtype"] = str(pil_img.mode)
            arr = np.array(pil_img)
            if arr.ndim == 2:
                raw_bands = [arr]
            else:
                raw_bands = [arr[:, :, i] for i in range(arr.shape[2])]
            metadata["count"] = len(raw_bands)
            metadata["height"] = arr.shape[0]
            metadata["width"] = arr.shape[1]

    num_channels = len(raw_bands)
    normalized_list = [normalize_band(b) for b in raw_bands]
    normalized_data = np.stack(normalized_list, axis=0)  # [C, H, W]

    # Generate viewable preview RGB image [H, W, 3] uint8
    if num_channels == 1:
        # Replicate panchromatic to 3 channels
        mono = (normalized_data[0] * 255.0).astype(np.uint8)
        preview_rgb = np.stack([mono, mono, mono], axis=-1)
    elif num_channels >= 3:
        # Extract R, G, B
        r = (normalized_data[0] * 255.0).astype(np.uint8)
        g = (normalized_data[1] * 255.0).astype(np.uint8)
        b = (normalized_data[2] * 255.0).astype(np.uint8)
        preview_rgb = np.stack([r, g, b], axis=-1)
    else:
        mono = (normalized_data[0] * 255.0).astype(np.uint8)
        preview_rgb = np.stack([mono, mono, mono], axis=-1)

    return normalized_data, metadata, preview_rgb


def image_to_tensor(normalized_data: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """
    Converts [C, H, W] normalized float32 numpy array to [1, C, H, W] torch.Tensor.
    """
    tensor = torch.from_numpy(normalized_data).unsqueeze(0).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def tensor_to_images(
    sr_tensor: torch.Tensor
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Converts super-resolved tensor [1, C, H, W] in [0.0, 1.0] back to:
    - viewable_rgb: [H, W, 3] uint8 True Color composite
    - viewable_cir: [H, W, 3] uint8 False Color Infrared composite (if 4-band RGB+NIR)
    """
    sr_np = sr_tensor.squeeze(0).detach().cpu().numpy()
    sr_np = np.clip(sr_np, 0.0, 1.0)
    c, h, w = sr_np.shape

    cir_composite = None

    if c == 1:
        mono = (sr_np[0] * 255.0).astype(np.uint8)
        rgb_composite = np.stack([mono, mono, mono], axis=-1)
    elif c == 3:
        r = (sr_np[0] * 255.0).astype(np.uint8)
        g = (sr_np[1] * 255.0).astype(np.uint8)
        b = (sr_np[2] * 255.0).astype(np.uint8)
        rgb_composite = np.stack([r, g, b], axis=-1)
    elif c >= 4:
        # 4-band: [R, G, B, NIR]
        r = (sr_np[0] * 255.0).astype(np.uint8)
        g = (sr_np[1] * 255.0).astype(np.uint8)
        b = (sr_np[2] * 255.0).astype(np.uint8)
        nir = (sr_np[3] * 255.0).astype(np.uint8)
        rgb_composite = np.stack([r, g, b], axis=-1)

        # False-Color Infrared (CIR): NIR, Red, Green
        # Highlights vegetation in vivid red and water in dark tones
        cir_composite = np.stack([nir, r, g], axis=-1)
    else:
        mono = (sr_np[0] * 255.0).astype(np.uint8)
        rgb_composite = np.stack([mono, mono, mono], axis=-1)

    return rgb_composite, cir_composite


def generate_uncertainty_heatmap(
    uncertainty_tensor: torch.Tensor,
    colormap_name: str = "turbo"
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generates a scientific epistemic uncertainty heatmap with colorbar and statistics.

    Args:
        uncertainty_tensor: [1, 1, H, W] standard deviation from MC Dropout.
        colormap_name: Matplotlib colormap ('turbo', 'magma', 'inferno', 'viridis').

    Returns:
        heatmap_image: [H, W, 3] uint8 RGB image of the rendered figure.
        stats: Dict of summary statistics (mean, max, std, p95).
    """
    unc_np = uncertainty_tensor.squeeze().detach().cpu().numpy()
    unc_np = np.nan_to_num(unc_np, nan=0.0)

    stats = {
        "mean_uncertainty": float(np.mean(unc_np)),
        "max_uncertainty": float(np.max(unc_np)),
        "std_uncertainty": float(np.std(unc_np)),
        "p95_uncertainty": float(np.percentile(unc_np, 95))
    }

    # Render publication-quality figure with colorbar
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(unc_np, cmap=colormap_name)
    ax.set_title("Epistemic Uncertainty Heatmap (15-Pass MC Dropout)", fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predictive Std Deviation (\u03c3)", rotation=270, labelpad=15, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    # Convert matplotlib figure to uint8 RGB numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    heatmap_img = np.array(pil_img)

    return heatmap_img, stats


def calculate_metrics(
    lr_preview: np.ndarray,
    sr_rgb: np.ndarray
) -> Dict[str, float]:
    """
    Computes objective quality and sharpness metrics:
    - PSNR against Bicubic reference
    - SSIM against Bicubic reference
    - Tenengrad / Laplacian variance sharpness score for both LR and SR
    - Sharpness enhancement factor
    """
    h_sr, w_sr = sr_rgb.shape[:2]

    # Generate bicubic upsampled baseline of low-res input for fair evaluation
    bicubic_baseline = cv2.resize(lr_preview, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)

    # Compute PSNR & SSIM
    psnr_val = float(compute_psnr(bicubic_baseline, sr_rgb, data_range=255))
    ssim_val = float(compute_ssim(bicubic_baseline, sr_rgb, channel_axis=2, data_range=255))

    # Compute Laplacian sharpness (Variance of Laplacian operator)
    gray_lr_up = cv2.cvtColor(bicubic_baseline, cv2.COLOR_RGB2GRAY)
    gray_sr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2GRAY)

    sharpness_lr = float(cv2.Laplacian(gray_lr_up, cv2.CV_64F).var())
    sharpness_sr = float(cv2.Laplacian(gray_sr, cv2.CV_64F).var())
    sharpness_gain = float(sharpness_sr / max(sharpness_lr, 1e-5))

    return {
        "psnr_db": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "sharpness_lr": round(sharpness_lr, 1),
        "sharpness_sr": round(sharpness_sr, 1),
        "sharpness_gain_ratio": round(sharpness_gain, 2)
    }


def create_side_by_side_banner(lr_preview: np.ndarray, sr_rgb: np.ndarray) -> np.ndarray:
    """Creates a side-by-side comparison image with clean title banner."""
    h_sr, w_sr = sr_rgb.shape[:2]
    lr_matched = cv2.resize(lr_preview, (w_sr, h_sr), interpolation=cv2.INTER_NEAREST)
    divider = np.full((h_sr, 3, 3), 39, dtype=np.uint8)  # Zinc-800 divider
    combined = np.hstack([lr_matched, divider, sr_rgb])

    banner_h = 32
    banner = np.full((banner_h, combined.shape[1], 3), 24, dtype=np.uint8)  # Dark zinc-900
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(banner, "INPUT (1x)", (14, 21), font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(banner, "BHUVISTAAR 2x RESIDUAL SR", (w_sr + 16, 21), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return np.vstack([banner, combined])


def save_geotiff(
    output_path: str,
    sr_tensor: torch.Tensor,
    original_metadata: Dict[str, Any]
) -> str:
    """
    Saves super-resolved multi-band data as a georeferenced GeoTIFF
    with updated 2x spatial transform if georeferencing exists.
    """
    sr_np = sr_tensor.squeeze(0).detach().cpu().numpy()  # [C, 2H, 2W]
    c, h, w = sr_np.shape

    if HAS_RASTERIO and original_metadata.get("is_geotiff") and original_metadata.get("transform"):
        # Scale affine transform for 2x resolution (pixel size halved)
        old_transform = original_metadata["transform"]
        new_transform = old_transform * Affine.scale(0.5, 0.5)

        # Scale float [0, 1] to original dtype range
        orig_dtype = original_metadata.get("dtype", "uint8")
        if "int16" in orig_dtype:
            out_data = (sr_np * 10000.0).astype(np.uint16)
            dst_dtype = "uint16"
        else:
            out_data = (sr_np * 255.0).astype(np.uint8)
            dst_dtype = "uint8"

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=c,
            dtype=dst_dtype,
            crs=original_metadata.get("crs"),
            transform=new_transform
        ) as dst:
            for band_idx in range(c):
                dst.write(out_data[band_idx], band_idx + 1)
    else:
        # Standard TIFF output via OpenCV or PIL
        out_uint8 = (sr_np * 255.0).astype(np.uint8)
        if c >= 3:
            rgb = np.transpose(out_uint8[:3], (1, 2, 0))
            Image.fromarray(rgb).save(output_path)
        else:
            Image.fromarray(out_uint8[0]).save(output_path)

    return output_path
