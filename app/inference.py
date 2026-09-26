"""
BhuVistaar - Standalone Satellite Inference Pipeline.
Provides high-performance 4x super-resolution and Monte Carlo epistemic uncertainty quantification
for programmatic workflows, scripts, and command-line execution.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional

import torch
import cv2
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import settings
from app.model import BhuVistaarModelManager
from app.utils import (
    load_satellite_image,
    image_to_tensor,
    tensor_to_images,
    generate_uncertainty_heatmap,
    calculate_metrics,
    create_side_by_side_banner,
    save_geotiff
)

# Global singleton model manager for zero redundant reloads
_GLOBAL_MODEL_MANAGER: Optional[BhuVistaarModelManager] = None


def get_model_manager(device: Optional[str] = None, weights_path: Optional[str] = None) -> BhuVistaarModelManager:
    """Returns singleton model manager, loading model only once."""
    global _GLOBAL_MODEL_MANAGER
    if _GLOBAL_MODEL_MANAGER is None:
        torch_device = torch.device(device) if device else None
        _GLOBAL_MODEL_MANAGER = BhuVistaarModelManager(device=torch_device, weights_path=weights_path)
    return _GLOBAL_MODEL_MANAGER


@torch.no_grad()
def run_super_resolution(
    input_path: str,
    output_dir: Optional[str] = None,
    num_passes: int = 5,
    colormap: str = "turbo",
    device: Optional[str] = None,
    weights_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes 4x super-resolution and Monte Carlo uncertainty mapping on a satellite image.
    
    Args:
        input_path: Path to input GeoTIFF or standard image.
        output_dir: Directory to save generated assets (defaults to settings.TEMP_OUTPUT_DIR).
        num_passes: Number of Monte Carlo stochastic passes for uncertainty quantification (default: 5).
        colormap: Palette for uncertainty heatmap ('turbo' or 'magma').
        device: Device to use ('cuda' or 'cpu').
        weights_path: Optional path to custom model weights.
        
    Returns:
        Dictionary containing quantitative metrics, execution timing, and output file paths.
    """
    start_time = time.time()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    target_output_dir = output_dir if output_dir else str(settings.TEMP_OUTPUT_DIR)
    os.makedirs(target_output_dir, exist_ok=True)

    manager = get_model_manager(device=device, weights_path=weights_path)

    # 1. Ingestion & radiometric normalization with memory cap
    normalized_data, metadata, lr_preview = load_satellite_image(input_path)
    orig_name = os.path.basename(input_path)
    metadata["file_name"] = orig_name

    # 2. Monte Carlo Dropout Super-Resolution
    effective_passes = min(num_passes, 5) if manager.device.type == "cpu" else num_passes
    tensor = image_to_tensor(normalized_data, device=manager.device)
    sr_tensor, unc_tensor = manager.super_resolve(tensor, num_passes=effective_passes)

    # 3. Compositing & Uncertainty Heatmap
    sr_rgb, sr_cir = tensor_to_images(sr_tensor)
    side_by_side = create_side_by_side_banner(lr_preview, sr_rgb)
    uncertainty_heatmap, unc_stats = generate_uncertainty_heatmap(unc_tensor, colormap_name=colormap)

    # 4. Quantitative Metrics
    metrics = calculate_metrics(
        lr_preview,
        sr_rgb,
        normalized_data=normalized_data,
        sr_tensor=sr_tensor
    )

    # 5. Export enhanced imagery to disk
    base_stem = os.path.splitext(orig_name)[0]
    out_ext = ".tif" if metadata.get("is_geotiff") else ".png"
    out_sr_path = os.path.join(target_output_dir, f"{base_stem}_SR_4x{out_ext}")
    save_geotiff(out_sr_path, sr_tensor, metadata)

    out_sbs_path = os.path.join(target_output_dir, f"{base_stem}_comparison.png")
    cv2.imwrite(out_sbs_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

    out_unc_path = os.path.join(target_output_dir, f"{base_stem}_uncertainty.png")
    cv2.imwrite(out_unc_path, cv2.cvtColor(uncertainty_heatmap, cv2.COLOR_RGB2BGR))

    elapsed_sec = round(time.time() - start_time, 2)

    # Memory cleanup
    del tensor, sr_tensor, unc_tensor, normalized_data
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "success": True,
        "input_path": input_path,
        "output_sr_path": out_sr_path,
        "output_comparison_path": out_sbs_path,
        "output_uncertainty_path": out_unc_path,
        "elapsed_seconds": elapsed_sec,
        "metrics": metrics,
        "uncertainty_stats": unc_stats,
        "dimensions": {
            "input": [int(lr_preview.shape[1]), int(lr_preview.shape[0])],
            "output": [int(sr_rgb.shape[1]), int(sr_rgb.shape[0])]
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BhuVistaar 4x Satellite Super-Resolution & Uncertainty CLI")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input satellite image (GeoTIFF/PNG)")
    parser.add_argument("-o", "--output_dir", type=str, default="data/cache/outputs", help="Output directory")
    parser.add_argument("-p", "--passes", type=int, default=5, help="Monte Carlo stochastic passes (default: 5)")
    parser.add_argument("-c", "--colormap", type=str, default="turbo", choices=["turbo", "magma"], help="Uncertainty colormap")
    parser.add_argument("-d", "--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    result = run_super_resolution(
        input_path=args.input,
        output_dir=args.output_dir,
        num_passes=args.passes,
        colormap=args.colormap,
        device=args.device
    )

    print(f"\n[BhuVistaar] Super-Resolution Finished in {result['elapsed_seconds']}s:")
    print(f"  Input        : {result['input_path']} ({result['dimensions']['input'][0]}x{result['dimensions']['input'][1]})")
    print(f"  Enhanced 4x  : {result['output_sr_path']} ({result['dimensions']['output'][0]}x{result['dimensions']['output'][1]})")
    print(f"  PSNR         : {result['metrics']['psnr_db']} dB")
    print(f"  SSIM         : {result['metrics']['ssim']}")
    print(f"  Uncertainty  : Mean={result['uncertainty_stats']['mean_uncertainty']:.4f}")
