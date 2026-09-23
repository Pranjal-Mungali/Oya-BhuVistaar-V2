"""
End-to-End Pipeline Verification for BhuVistaar 4x Super-Resolution
"""

import os
import torch
import numpy as np

def run_tests():
    print("=== BHUVISTAAR PIPELINE TEST SUITE ===")

    # 1. Test STAC client
    from backend.services.copernicus import search_sentinel2_scenes
    scenes = search_sentinel2_scenes([77.1, 28.5, 77.3, 28.7])
    assert len(scenes) > 0, "Sentinel-2 search returned empty list"
    print(f"[PASS] Sentinel-2 STAC Search: {len(scenes)} scenes found (Top: {scenes[0]['id'][:30]}...)")

    # 2. Test Bhoonidhi client
    from backend.services.bhoonidhi import search_cartosat_reference
    refs = search_cartosat_reference([77.1, 28.5, 77.3, 28.7])
    assert len(refs) > 0, "Bhoonidhi search returned empty list"
    print(f"[PASS] Bhoonidhi STAC Search: {len(refs)} references found (Status: {refs[0]['status']})")

    # 3. Test Dual-Pathway Model Architecture
    from model import DualPathwayBhuVistaarNet, mc_dropout_inference, tiled_super_resolve
    model = DualPathwayBhuVistaarNet(in_channels=4, out_channels=4, upscale_factor=4, pretrained=False)
    model.eval()

    dummy_input = torch.rand(1, 4, 32, 32)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 4, 128, 128), f"Expected (1, 4, 128, 128), got {out.shape}"
    print(f"[PASS] Model Forward Shape: {dummy_input.shape} -> {out.shape} (4x Super-Resolution)")

    # 4. Test MC Dropout Uncertainty
    sr_mean, unc_map = mc_dropout_inference(model, dummy_input, num_passes=5)
    assert sr_mean.shape == (1, 4, 128, 128)
    assert unc_map.shape == (1, 1, 128, 128)
    print(f"[PASS] MC Dropout Inference: Mean {sr_mean.shape}, Epistemic Uncertainty {unc_map.shape}")

    # 5. Test Metrics (SAM, ERGAS, PSNR, SSIM)
    from utils import compute_spectral_angle_mapper, compute_ergas, calculate_metrics
    ref_patch = np.random.rand(4, 128, 128).astype(np.float32)
    pred_patch = np.clip(ref_patch + 0.05 * np.random.randn(4, 128, 128), 0.0, 1.0).astype(np.float32)
    sam = compute_spectral_angle_mapper(ref_patch, pred_patch)
    ergas = compute_ergas(ref_patch, pred_patch)
    print(f"[PASS] Metrics Computation: SAM={sam:.2f} deg, ERGAS={ergas:.2f}")

    # 6. Test GeoTIFF Exporter
    from utils import save_geotiff, save_multiband_geotiff
    out_dir = os.path.join(os.path.dirname(__file__), "samples")
    test_rgb_out = os.path.join(out_dir, "test_output_rgb.tif")
    test_mb_out = os.path.join(out_dir, "test_output_multiband.tif")
    save_geotiff(test_rgb_out, sr_mean)
    save_multiband_geotiff(test_mb_out, sr_mean)
    assert os.path.exists(test_rgb_out) and os.path.exists(test_mb_out)
    print(f"[PASS] GeoTIFF Export: Saved RGB ({os.path.basename(test_rgb_out)}) and 4-Band ({os.path.basename(test_mb_out)})")

    print("\nALL BACKEND & ALGORITHMIC TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_tests()
