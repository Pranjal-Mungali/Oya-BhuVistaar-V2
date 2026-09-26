<div align="center">

<img src="assets/logo.png" alt="BhuVistaar Logo" width="160" />

# 🛰️ BhuVistaar V2
### AI-Powered Satellite 4× Super-Resolution & Bayesian Epistemic Uncertainty Mapping

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.0%2B-black?style=flat&logo=next.js&logoColor=white)](https://nextjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4%2B-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-NVIDIA_RTX-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>High-fidelity 4× sub-pixel super-resolution for Earth observation imagery (10m &rarr; 2.5m GSD) with Monte Carlo Dropout epistemic uncertainty quantification & georeferenced GeoTIFF export.</b>
</p>

[Completed Features](#-completed-features) •
[Dashboard Preview](#-dashboard-preview) •
[Visual Results](#-visual-results) •
[System Architecture](#-system-architecture) •
[Quick Start](#-quick-start) •
[Roadmap](#-upcoming-features-more-features-coming-soon)

</div>

---

## 📌 Overview

**BhuVistaar** is an AI-powered super-resolution platform designed for Earth observation satellites (such as ESA Sentinel-2). It enhances **10m low-resolution multispectral imagery to 2.5m ground sampling distance (GSD)** while calculating Bayesian epistemic uncertainty to prevent hallucinations and flag ambiguous terrain boundaries.

---

## ✅ Completed Features

- **4× Deep Super-Resolution**: Powered by a 23-block RRDBNet (Real-ESRGAN backbone) reconstructing sharp building footprints, road networks, and natural textures.
- **Monte Carlo Epistemic Uncertainty Mapping**: Executes 5–30 stochastic passes with spatial dropout to quantify model confidence and render calibrated uncertainty heatmaps ($\sigma$).
- **Interactive Dark Dashboard (Next.js 14)**:
  - **Curtain Wipe View**: Real-time slider comparing 10m LR input with 2.5m 4× SR output, with zoom ($1\times$ to $4\times$), pan & drag, and reset.
  - **Side-by-Side Dual Viewport**: Synchronized comparative analysis across full scene extents.
  - **Uncertainty Heatmap View**: Scientific colormaps (**Turbo**, **Magma**, **Inferno**, **Viridis**) with calibrated scale.
  - **False-Color CIR Composite**: Near-Infrared (NIR) vegetation composite (B8 + B4 + B3).
  - **Real-time Quality Telemetry**: Live PSNR, SSIM, Tenengrad Laplacian Sharpness Gain, and mean uncertainty ($\sigma$).
  - **Multispectral Band Telemetry**: Interactive area chart tracking fidelity across Sentinel-2 bands.
- **FastAPI High-Performance Backend**:
  - Asynchronous single-click inference endpoints (`/api/super-resolve`, `/api/predict`).
  - Pre-packaged benchmark sample catalog (`/api/samples`).
- **Georeferenced 2.5m GeoTIFF Export**:
  - Automatically scales the affine transformation matrix ($\mathbf{A}_{new} = \mathbf{A}_{old} \times 0.25$).
  - Preserves Coordinate Reference System (CRS) for direct drag-and-drop into **QGIS** and **ArcGIS Pro**.
- **PyTorch Training Pipeline (`train.py`)**:
  - L1 Reconstruction Loss + VGG-19 Perceptual Loss.
  - Cosine Annealing learning rate scheduling and data augmentation.
  - Multi-metric validation (PSNR & SSIM tracking) with automatic best-checkpoint saving.

---

## 🖥️ Dashboard Preview

![BhuVistaar Dashboard Studio](docs/screenshots/dashboard_full.png)

### Interactive Curtain Wipe (10m vs 2.5m GSD):
![Curtain Wipe 4x Resolution Inspection](docs/screenshots/dashboard_overview.png)

---

## 🔬 Visual Results

### 1. 4× Super-Resolution Comparison (10m &rarr; 2.5m)
BhuVistaar resolves individual buildings, roads, and land boundaries with preserved radiometry:

| Original 10m Ground Sample (1× Input) | BhuVistaar 2.5m Super-Resolved (4× Output) |
| :---: | :---: |
| ![Original 10m Sample](docs/screenshots/sample_input_1x.png) | ![4x Super-Resolved 2.5m](docs/screenshots/sample_super_res_4x.png) |

### 2. Bayesian Epistemic Uncertainty Heatmap ($\sigma$)
Quantifies prediction variance across 15 stochastic dropout passes to highlight ambiguous textures and shadow edges:

![Epistemic Uncertainty Heatmap](docs/screenshots/epistemic_uncertainty_map.png)

### 3. False-Color Infrared (CIR: NIR + Red + Green)
Generates high-contrast vegetation composites for agricultural canopy and water boundary delineation:

![False Color NIR Composite](docs/screenshots/false_color_nir.png)

---

## 🏗️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             1. FRONTEND DASHBOARD (Next.js 14)                         │
│   • Curtain Wipe Slider (1x vs 4x)           • Side-by-Side Dual Viewport              │
│   • Epistemic Uncertainty Heatmap Viewer     • False-Color NIR Composite Analyzer      │
│   • Live Telemetry (PSNR, SSIM, Sharpness)   • 1-Click 2.5m GeoTIFF Download           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ HTTP / REST (Multipart + JSON)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              2. BACKEND API (FastAPI)                                  │
│   • POST /api/super-resolve : 1-Click 4x Real-ESRGAN inference                         │
│   • POST /api/predict       : 4x Super-Resolution + Monte Carlo uncertainty passes     │
│   • GET  /api/samples       : Built-in Sentinel-2 L2A test catalog                     │
│   • GET  /api/download/{id} : Georeferenced GeoTIFF & High-DPI PNG streaming           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        3. RADIOMETRIC INGESTION & NORMALIZATION                        │
│   • Rasterio 16-bit GeoTIFF reader & Affine Geotransform extractor                     │
│   • Multi-spectral demux: RGB (3-Band) vs RGB+NIR (4-Band)                             │
│   • Percentile dynamic range normalization [p2, p98] -> Float32 [0.0, 1.0]             │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Normalized Tensor [1, C, H, W]
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      4. DEEP SUPER-RESOLUTION & UNCERTAINTY CORE                       │
│   • 23-Block RRDBNet (Real-ESRGAN) with 64-channel deep residual trunk                 │
│   • Test-time Monte Carlo Spatial Dropout (p = 0.06)                                   │
│   • T Stochastic passes -> Predictive Mean & Epistemic Variance (σ^2)                  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         5. GIS GEOREFERENCING & SPATIAL EXPORT                         │
│   • Scaled Affine transform: A_new = A_old * Scale(0.25, 0.25)                         │
│   • Native 2.5m GeoTIFF output with preserved CRS (EPSG:4326 / UTM)                    │
│   • Direct drag-and-drop into QGIS & ArcGIS Pro                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Layout

```text
BhuVistaar/
├── app/                            # Core Application Package
│   ├── main.py                     # FastAPI production server & API routing
│   ├── model.py                    # 4x RRDBNet architecture & MC Dropout engine
│   ├── utils.py                    # GeoTIFF I/O, radiometric normalization, heatmaps
│   ├── inference.py                # Standalone programmatic & CLI inference pipeline
│   └── services/                   # STAC providers (Copernicus, Bhoonidhi)
├── training/                       # Deep Learning Training Pipeline
│   ├── train.py                    # Model training loop & validation orchestration
│   ├── dataset.py                  # SatelliteCropDataset & data augmentations
│   └── losses.py                   # L1, VGG-19 perceptual, Sobel edge, & SSIM/PSNR
├── frontend/                       # Next.js 14 + Tailwind CSS Dashboard
│   ├── app/dashboard/page.tsx      # Main studio dashboard view
│   ├── components/                 # SatelliteStudio, SectionCards, charts
│   └── package.json
├── weights/                        # Model weights (RealESRGAN_x4plus.pth)
├── samples/                        # Pre-packaged Sentinel-2 test scenes
├── assets/                         # Branding & visual assets
│   ├── logo.png
│   ├── logo-dark.png
│   └── screenshots/
├── docs/                           # Documentation & architecture specifications
├── scripts/                        # Automation & testing utilities
│   ├── dev.py                      # Multi-server development runner
│   ├── run_dev.bat                 # Windows one-click launcher
│   ├── create_samples.py           # Sample generator
│   └── test_pipeline.py            # End-to-end integration test suite
├── .gitignore
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Production multi-stage container
├── README.md
└── render.yaml                     # Cloud deployment blueprint
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python**: `3.10` or higher
- **Node.js**: `18.x` or higher
- **NVIDIA GPU** *(Optional)*: CUDA 11.8+ for hardware acceleration (CPU fallback included)

### 2. Single-Command Launch (Backend + Frontend Concurrently)

You can launch both the FastAPI backend and Next.js frontend concurrently with a **single command**:

```bash
# Option A: Standard Monorepo NPM Command (Cross-platform)
npm run dev

# Option B: Pure Python Unified Runner (Cross-platform)
python scripts/dev.py

# Option C: Windows One-Click Batch Launcher
scripts\run_dev.bat
```

- **Frontend Dashboard**: [`http://localhost:3000`](http://localhost:3000)
- **FastAPI Backend**: [`http://127.0.0.1:8000`](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)

---

### 3. Deploying to Render (Cloud Deployment)

BhuVistaar V2 features a **multi-stage production Dockerfile** that builds the Next.js frontend into static assets and serves both the API and the interactive UI from FastAPI on a single dynamic `$PORT`.

#### Option A: 1-Click Render Blueprint (Recommended)
1. Push your repository to GitHub.
2. In [Render Dashboard](https://dashboard.render.com/), click **New +** -> **Blueprint**.
3. Select this repository. Render automatically reads [`render.yaml`](file:///d:/BhuVistaar%20V2/render.yaml) and configures the web service.
4. Click **Apply** to deploy!

#### Option B: Manual Web Service Setup
1. Click **New +** -> **Web Service** -> **Deploy existing image or repository**.
2. Select your repository.
3. Choose **Docker** as the Environment.
4. Render will build the multi-stage Docker container and automatically expose the service on `$PORT`.

---

## 🗺️ GIS Integration (QGIS & ArcGIS)

1. Super-resolve any satellite scene in the dashboard.
2. Click **Download 4× Enhanced Image** (exports `.tif` with 2.5m GSD).
3. Drag and drop the downloaded file directly into **QGIS** or **ArcGIS Pro**.
4. The enhanced image automatically aligns to the exact coordinates and vector layers with $4\times$ spatial density.

---

## 🔮 Upcoming Features (More Features Coming Soon)

We are actively developing the following capabilities for upcoming releases:

- [ ] **Live Satellite STAC Integration**: Direct connection to Copernicus Data Space & Microsoft Planetary Computer for on-the-fly Sentinel-2 tile fetching.
- [ ] **ISRO Bhoonidhi STAC Support**: Automated pairing with Cartosat high-resolution references over the Indian subcontinent.
- [ ] **Interactive Slippy Map AOI Selector**: MapLibre / Leaflet bounding-box selector to draw areas of interest on live global satellite basemaps.
- [ ] **TensorRT & ONNX Runtime Optimization**: FP16 / INT8 quantized neural acceleration for sub-second inference on large regional scenes.
- [ ] **Temporal Change Detection**: Multi-date super-resolution comparison to monitor urban growth, deforestation, and flood inundation.
- [ ] **Automated Cloud Masking**: SCL (Scene Classification Layer) band filtering to detect and handle cloud/shadow occlusion.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
