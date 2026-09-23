"""
BhuVistaar - FastAPI Production Backend Server.
Provides 4x Satellite Super-Resolution, Monte Carlo Dropout Epistemic Uncertainty Mapping,
Multispectral Spectral Telemetry, and GIS GeoTIFF Export.
"""

import os
import io
import time
import base64
import tempfile
import uuid
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, JSONResponse
import cv2
import numpy as np
from PIL import Image

from model import BhuVistaarModelManager
from utils import (
    load_satellite_image,
    image_to_tensor,
    tensor_to_images,
    generate_uncertainty_heatmap,
    calculate_metrics,
    create_side_by_side_banner,
    save_geotiff
)

app = FastAPI(
    title="BhuVistaar API",
    description="Satellite Super-Resolution & Epistemic Uncertainty Quantification (4x)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = BhuVistaarModelManager()

@app.on_event("startup")
def startup_warmup():
    """Initializes models and GPU context on server startup."""
    try:
        model_manager.warmup()
    except Exception as e:
        print(f"[BhuVistaar API] Warmup notice: {e}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
TEMP_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "bhuvistaar_outputs")
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)


def numpy_to_base64_png(arr: np.ndarray) -> str:
    """Converts uint8 image array to base64 data URI."""
    pil_img = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "project": "BhuVistaar",
        "version": "2.0.0",
        "device": str(model_manager.device),
        "cuda_available": str(model_manager.device) != "cpu"
    }


@app.get("/api/samples")
def get_samples():
    """Returns metadata for pre-packaged satellite test scenes."""
    samples = [
        {
            "id": "sentinel2_nir",
            "name": "Sentinel-2 Multispectral (RGB + NIR)",
            "filename": "sample_sentinel2_rgb_nir.tif",
            "type": "GeoTIFF (uint16)",
            "bands": 4,
            "bands_desc": "B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)",
            "crs": "EPSG:4326 (WGS 84)",
            "resolution": "128 x 128 px"
        },
        {
            "id": "sentinel2_rgb",
            "name": "Sentinel-2 True Color (RGB)",
            "filename": "sample_sentinel2_rgb.tif",
            "type": "GeoTIFF (uint16)",
            "bands": 3,
            "bands_desc": "B4 (Red), B3 (Green), B2 (Blue)",
            "crs": "EPSG:4326 (WGS 84)",
            "resolution": "128 x 128 px"
        }
    ]
    return {"samples": samples}


@app.post("/api/predict")
async def predict_satellite(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    num_passes: int = Form(15),
    colormap: str = Form("turbo")
):
    """Executes 4x super-resolution and Monte Carlo Dropout epistemic uncertainty quantification."""
    temp_input_path = None
    orig_name = "scene"

    try:
        if file is not None and file.filename:
            orig_name = file.filename
            file_ext = os.path.splitext(file.filename)[1]
            temp_input_path = os.path.join(TEMP_OUTPUT_DIR, f"upload_{uuid.uuid4().hex[:8]}{file_ext}")
            contents = await file.read()
            with open(temp_input_path, "wb") as f:
                f.write(contents)
        elif sample_id:
            sample_map = {
                "sentinel2_nir": "sample_sentinel2_rgb_nir.tif",
                "sentinel2_rgb": "sample_sentinel2_rgb.tif",
                "cartosat_urban": "sample_cartosat_urban.tif",
                "urban_optical": "sample_urban_lr.png"
            }
            if sample_id not in sample_map:
                raise HTTPException(status_code=400, detail=f"Unknown sample_id: {sample_id}")
            temp_input_path = os.path.join(SAMPLES_DIR, sample_map[sample_id])
            orig_name = sample_map[sample_id]
        else:
            # Default to bundled Sentinel-2 NIR sample
            temp_input_path = os.path.join(SAMPLES_DIR, "sample_sentinel2_rgb_nir.tif")
            orig_name = "sample_sentinel2_rgb_nir.tif"

        if not os.path.exists(temp_input_path):
            raise HTTPException(status_code=404, detail="Target satellite image not found")

        # Ingestion & radiometric normalization
        normalized_data, metadata, lr_preview = load_satellite_image(temp_input_path)
        metadata["file_name"] = orig_name

        # Monte Carlo Dropout Inference (stochastic passes)
        tensor = image_to_tensor(normalized_data, device=model_manager.device)
        sr_tensor, unc_tensor = model_manager.super_resolve(tensor, num_passes=num_passes)

        # Compositing & Uncertainty Heatmap Generation
        sr_rgb, sr_cir = tensor_to_images(sr_tensor)
        side_by_side = create_side_by_side_banner(lr_preview, sr_rgb)
        uncertainty_heatmap, unc_stats = generate_uncertainty_heatmap(unc_tensor, colormap_name=colormap)

        # Quantitative Metrics & Multispectral Analysis
        metrics = calculate_metrics(
            lr_preview,
            sr_rgb,
            normalized_data=normalized_data,
            sr_tensor=sr_tensor
        )

        # GeoTIFF Export (scaled affine transform)
        out_ext = ".tif" if metadata.get("is_geotiff") else ".png"
        out_id = uuid.uuid4().hex[:8]
        out_filename = f"BhuVistaar_SR_4x_{out_id}{out_ext}"
        out_file_path = os.path.join(TEMP_OUTPUT_DIR, out_filename)
        save_geotiff(out_file_path, sr_tensor, metadata)

        # Base64 Encoding for interactive UI
        lr_base64 = numpy_to_base64_png(lr_preview)
        sr_base64 = numpy_to_base64_png(sr_rgb)
        sbs_base64 = numpy_to_base64_png(side_by_side)
        unc_base64 = numpy_to_base64_png(uncertainty_heatmap)
        cir_base64 = numpy_to_base64_png(sr_cir) if sr_cir is not None else None

        return {
            "success": True,
            "images": {
                "low_res": lr_base64,
                "super_res": sr_base64,
                "side_by_side": sbs_base64,
                "uncertainty_heatmap": unc_base64,
                "false_color_cir": cir_base64
            },
            "metrics": {
                "psnr_db": metrics["psnr_db"],
                "ssim": metrics["ssim"],
                "sharpness_gain_ratio": metrics["sharpness_gain_ratio"],
                "sharpness_lr": metrics["sharpness_lr"],
                "sharpness_sr": metrics["sharpness_sr"],
                "mean_uncertainty": round(unc_stats["mean_uncertainty"], 5),
                "max_uncertainty": round(unc_stats["max_uncertainty"], 5),
                "std_uncertainty": round(unc_stats["std_uncertainty"], 5),
                "p95_uncertainty": round(unc_stats["p95_uncertainty"], 5),
                "spectral_bands": metrics.get("spectral_bands", [])
            },
            "metadata": {
                "file_name": metadata.get("file_name", "scene"),
                "crs": metadata.get("crs", "EPSG:4326 (WGS 84)"),
                "bands_count": int(normalized_data.shape[0]),
                "bit_depth": str(metadata.get("dtype", "uint16")),
                "is_geotiff": bool(metadata.get("is_geotiff")),
                "input_shape": [int(normalized_data.shape[1]), int(normalized_data.shape[2])],
                "output_shape": [int(sr_rgb.shape[0]), int(sr_rgb.shape[1])],
                "has_nir": bool(sr_cir is not None)
            },
            "download_url": f"/api/download/{out_filename}",
            "filename": out_filename
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# Endpoint alias for compatibility
@app.post("/api/super-resolution")
async def super_resolution_alias(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    num_passes: int = Form(15),
    colormap: str = Form("turbo")
):
    return await predict_satellite(file=file, sample_id=sample_id, num_passes=num_passes, colormap=colormap)


@app.get("/api/download/{filename}")
def download_asset(filename: str):
    file_path = os.path.join(TEMP_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found or expired")
    media_type = "image/tiff" if filename.endswith(".tif") else "image/png"
    return FileResponse(file_path, media_type=media_type, filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
