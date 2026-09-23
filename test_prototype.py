"""
Acceptance Test for Simplified BhuVistaar Prototype
Verifies:
1. Sample image exists and is readable (512x512)
2. Backend super-resolution endpoint executes Real-ESRGAN 4x inference
3. Output is exactly 4x dimensions (2048x2048)
4. Laplacian sharpness is significantly improved vs bicubic
5. Download endpoint serves the enhanced image correctly
"""

import os
from PIL import Image
import cv2
import numpy as np
from fastapi.testclient import TestClient
from server import app

def test_full_prototype():
    print("=== BHUVISTAAR PROTOTYPE ACCEPTANCE TEST ===")
    client = TestClient(app)

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health = res.json()
    print(f"[PASS] 1. API Health: {health['status']} | Model: {health['model']} | Device: {health['device']}")

    # 2. Sample metadata
    res = client.get("/api/sample")
    assert res.status_code == 200, f"Sample metadata failed: {res.text}"
    sample_meta = res.json()
    assert sample_meta["width"] == 512 and sample_meta["height"] == 512
    print(f"[PASS] 2. Bundled Sample Image: {sample_meta['width']}x{sample_meta['height']} ({sample_meta['resolution']})")

    # 3. Execute 4x Super-Resolution
    print("Executing 4x Super-Resolution inference...")
    res = client.post("/api/super-resolution")
    assert res.status_code == 200, f"SR inference failed: {res.text}"
    data = res.json()
    assert data["success"] is True
    assert data["scale"] == 4
    assert data["input_width"] == 512 and data["input_height"] == 512
    assert data["output_width"] == 2048 and data["output_height"] == 2048
    print(f"[PASS] 3. 4x Neural Inference: {data['input_width']}x{data['input_height']} -> {data['output_width']}x{data['output_height']} in {data['processing_time_sec']}s")

    # 4. Verify Download Asset
    dl_url = data["download_url"]
    dl_res = client.get(dl_url)
    assert dl_res.status_code == 200, f"Download failed: {dl_res.status_code}"
    assert len(dl_res.content) > 100000, "Downloaded file is suspiciously small"

    # Save and verify dimensions independently
    test_out = "samples/downloaded_test_output.png"
    with open(test_out, "wb") as f:
        f.write(dl_res.content)

    saved_im = Image.open(test_out)
    assert saved_im.size == (2048, 2048), f"Saved image size {saved_im.size} != (2048, 2048)"
    print(f"[PASS] 4. Downloaded Asset: Verified {saved_im.size} independently ({os.path.getsize(test_out)} bytes)")

    # 5. Visual enhancement verification
    lr_cv = cv2.imread("samples/satellite_sample.png")
    sr_cv = cv2.imread(test_out)
    bicubic_cv = cv2.resize(lr_cv, (2048, 2048), interpolation=cv2.INTER_CUBIC)

    lap_bicubic = cv2.Laplacian(cv2.cvtColor(bicubic_cv, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    lap_sr = cv2.Laplacian(cv2.cvtColor(sr_cv, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    ratio = lap_sr / max(lap_bicubic, 1e-5)

    print(f"[PASS] 5. Sharpness Verification: Bicubic={lap_bicubic:.1f}, Real-ESRGAN={lap_sr:.1f}, Factor={ratio:.2f}x")
    assert ratio > 2.0, "Model did not produce significant edge enhancement over bicubic"

    print("\nALL ACCEPTANCE TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_full_prototype()
