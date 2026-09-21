"""
BhuVistaar - FastAPI Backend Server

High-performance asynchronous API for satellite super-resolution,
Monte Carlo Dropout uncertainty quantification, and GeoTIFF georeferenced export.
"""

import os
import io
import base64
import tempfile
import uuid
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel
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
    save_geotiff
)

app = FastAPI(
    title="BhuVistaar API",
    description="Satellite Super-Resolution & Epistemic Uncertainty Quantification",
    version="2.0.0"
)

# Enable CORS for Next.js development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model Manager
model_manager = BhuVistaarModelManager()

@app.on_event("startup")
def startup_warmup():
    """Warms up CUDA device and caches model instances for instantaneous inference."""
    try:
        model_manager.warmup()
    except Exception as e:
        print(f"[BhuVistaar API] Warmup notice: {e}")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
TEMP_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "bhuvistaar_outputs")
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)


def numpy_to_base64_png(arr: np.ndarray) -> str:
    """Converts uint8 RGB numpy array to base64 data URI."""
    pil_img = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def create_side_by_side_banner(lr_preview: np.ndarray, sr_rgb: np.ndarray) -> np.ndarray:
    """Creates side-by-side comparison with clean banner."""
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


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BhuVistaar API</title>
        <style>
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: #09090b;
                color: #f4f4f5;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
            }
            .card {
                background-color: #141416;
                border: 1px solid #27272a;
                border-radius: 12px;
                padding: 32px;
                max-width: 540px;
                width: 90%;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }
            .badge {
                display: inline-block;
                background-color: #1c1c1f;
                color: #22c55e;
                border: 1px solid #27272a;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 10px;
                border-radius: 20px;
                margin-bottom: 12px;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
            h1 {
                font-size: 24px;
                font-weight: 700;
                margin: 0 0 6px 0;
                color: #ffffff;
            }
            p {
                font-size: 13px;
                color: #a1a1aa;
                line-height: 1.5;
                margin: 0 0 24px 0;
            }
            .buttons {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            .btn {
                display: flex;
                align-items: center;
                justify-content: center;
                text-decoration: none;
                font-size: 13px;
                font-weight: 600;
                padding: 12px 16px;
                border-radius: 8px;
                transition: all 0.15s ease;
            }
            .btn-primary {
                background-color: #ffffff;
                color: #09090b;
            }
            .btn-primary:hover {
                background-color: #e4e4e7;
            }
            .btn-secondary {
                background-color: #1c1c1f;
                color: #f4f4f5;
                border: 1px solid #27272a;
            }
            .btn-secondary:hover {
                background-color: #27272a;
            }
            .footer {
                margin-top: 20px;
                padding-top: 16px;
                border-top: 1px solid #1f1f23;
                font-size: 11px;
                color: #71717a;
                display: flex;
                justify-content: space-between;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <span class="badge">&#9679; API Server Online</span>
            <h1>BhuVistaar Backend</h1>
            <p>
                The PyTorch Super-Resolution & MC Dropout inference engine is running.
                Access the <strong>Next.js shadcn/ui Dashboard</strong> or test the API via Swagger documentation.
            </p>
            <div class="buttons">
                <a href="http://localhost:3000" class="btn btn-primary" target="_blank">
                    &rarr; Open Next.js Dashboard (http://localhost:3000)
                </a>
                <a href="/docs" class="btn btn-secondary">
                    View Swagger API Documentation (/docs)
                </a>
            </div>
            <div class="footer">
                <span>AI Super Resolution</span>
                <span>BhuVistaar v2.0</span>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "project": "BhuVistaar",
        "version": "2.0.0",
        "device": str(model_manager.device),
        "pytorch_version": "2.12.1"
    }


@app.get("/api/samples")
def get_samples():
    """Returns metadata and preview thumbnails of pre-packaged test scenes."""
    samples = [
        {
            "id": "sentinel2_rgb",
            "name": "Sentinel-2 True Color (RGB)",
            "filename": "sample_sentinel2_rgb.tif",
            "type": "GeoTIFF (uint16)",
            "bands": 3,
            "bands_desc": "B4 (Red), B3 (Green), B2 (Blue)",
            "crs": "EPSG:4326 (WGS 84)",
            "resolution": "128 x 128 px"
        },
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
            "id": "urban_optical",
            "name": "Urban Optical Scene",
            "filename": "sample_urban_lr.png",
            "type": "PNG (uint8)",
            "bands": 3,
            "bands_desc": "Red, Green, Blue",
            "crs": "Non-projected",
            "resolution": "128 x 128 px"
        }
    ]
    return {"samples": samples}


@app.post("/api/predict")
def predict_satellite(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    num_passes: int = Form(15),
    colormap: str = Form("turbo")
):
    """
    Core inference pipeline:
    Loads satellite file, executes 2x SR with Monte Carlo Dropout,
    computes epistemic uncertainty, calculates metrics, and returns JSON payload.
    """
    temp_input_path = None

    try:
        if file is not None and file.filename:
            file_ext = os.path.splitext(file.filename)[1]
            temp_input_path = os.path.join(TEMP_OUTPUT_DIR, f"upload_{uuid.uuid4().hex[:8]}{file_ext}")
            contents = file.file.read()
            with open(temp_input_path, "wb") as f:
                f.write(contents)
        elif sample_id:
            sample_map = {
                "sentinel2_rgb": "sample_sentinel2_rgb.tif",
                "sentinel2_nir": "sample_sentinel2_rgb_nir.tif",
                "urban_optical": "sample_urban_lr.png"
            }
            if sample_id not in sample_map:
                raise HTTPException(status_code=400, detail=f"Unknown sample_id: {sample_id}")
            temp_input_path = os.path.join(SAMPLES_DIR, sample_map[sample_id])
        else:
            raise HTTPException(status_code=400, detail="Must provide either an uploaded file or a sample_id")

        if not os.path.exists(temp_input_path):
            raise HTTPException(status_code=404, detail="Target satellite image not found")

        # 1. Load satellite image
        normalized_data, metadata, lr_preview = load_satellite_image(temp_input_path)

        # 2. Tensor conversion & Inference
        tensor = image_to_tensor(normalized_data, device=model_manager.device)
        sr_tensor, unc_tensor = model_manager.super_resolve(tensor, num_passes=num_passes)

        # 3. Viewable composites
        sr_rgb, sr_cir = tensor_to_images(sr_tensor)
        side_by_side = create_side_by_side_banner(lr_preview, sr_rgb)
        uncertainty_heatmap, unc_stats = generate_uncertainty_heatmap(unc_tensor, colormap_name=colormap)

        # 4. Metrics
        metrics = calculate_metrics(lr_preview, sr_rgb)

        # 5. Export GeoTIFF for download
        out_ext = ".tif" if metadata.get("is_geotiff") else ".png"
        out_id = uuid.uuid4().hex[:8]
        out_filename = f"BhuVistaar_SR_2x_{out_id}{out_ext}"
        out_file_path = os.path.join(TEMP_OUTPUT_DIR, out_filename)
        save_geotiff(out_file_path, sr_tensor, metadata)

        # 6. Encode images as base64 data URIs
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
                "p95_uncertainty": round(unc_stats["p95_uncertainty"], 5)
            },
            "metadata": {
                "file_name": metadata.get("file_name", "scene"),
                "crs": metadata.get("crs", "Non-projected / Local"),
                "bands_count": int(normalized_data.shape[0]),
                "bit_depth": str(metadata.get("dtype", "uint8")),
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
