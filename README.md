# 🛰️ BhuVistaar: AI-Powered Satellite Super-Resolution & Uncertainty Mapping

**Smart India Hackathon 2026** | **Team:** The Outliers  
**Domain:** Space Technology / Disaster Management / Precision Agriculture / Geospatial Intelligence

---

## 📌 Executive Summary

Modern Earth observation satellites (e.g., Sentinel-2 at 10m/20m resolution, Landsat 8/9 at 30m resolution) provide high-frequency multispectral coverage of our planet. However, critical applications such as disaster assessment, illegal encroachment detection, crop disease delineation, and hydrological planning demand sub-meter or enhanced spatial detail.

**BhuVistaar** is an AI-driven, multi-band super-resolution platform that performs **2x spatial enhancement** with **Bayesian Epistemic Uncertainty Estimation** via Monte Carlo Dropout. Designed specifically for Earth observation pipelines, BhuVistaar natively supports high-bit-depth GeoTIFFs (`uint16`, `int16`, `float32`), true-color RGB, and multi-spectral Near-Infrared (NIR) bands while preserving spatial georeferencing coordinates and transforms.

---

## 🌟 Key Features

1. **Sub-Pixel Residual Super-Resolution (2x)**:
   - Deep Residual Convolutional Neural Network with **PixelShuffle** (Sub-pixel convolution).
   - Global residual skip connection prevents spatial artifacts and maintains photometric fidelity.

2. **Multispectral & Multi-Band Ingestion**:
   - Handles **3-band RGB** and **4-band RGB + NIR** (Sentinel-2, Landsat-8/9, PlanetScope).
   - Color Infrared (CIR: NIR + Red + Green) composite generation highlighting vegetation health and water boundaries.

3. **Epistemic Uncertainty Estimation (Monte Carlo Dropout)**:
   - Performs **15 stochastic forward passes** under active spatial dropout.
   - Computes predictive mean (super-resolved image) and predictive standard deviation ($\sigma$) per pixel.
   - Produces scientific uncertainty heatmaps with customizable color scales (`turbo`, `magma`, `inferno`, `viridis`).

4. **Robust Remote Sensing Preprocessing**:
   - Natively reads `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg` via **Rasterio** and **OpenCV**.
   - Adaptive percentile contrast-stretching (`[p1, p99]`) preventing the common "dark/black image" issue with 16-bit satellite reflectances.

5. **Scientific Evaluation Metrics**:
   - **PSNR (Peak Signal-to-Noise Ratio)**: Quantifies reconstruction fidelity.
   - **SSIM (Structural Similarity Index)**: Measures structural and edge consistency.
   - **Laplacian Variance Sharpness**: Measures high-frequency gradient enhancement ($1.1\times - 1.3\times$ gain).

6. **GeoTIFF Georeference Preservation**:
   - Preserves CRS (Coordinate Reference System, e.g., EPSG:4326, UTM) and updates the Affine transform ($2\times$ pixel density) for direct integration into QGIS and ArcGIS.

---

## 📂 Project Structure

```text
BhuVistaar/
├── frontend/                       # Next.js + React + shadcn/ui Dashboard (dashboard-01)
│   ├── app/
│   │   ├── dashboard/page.tsx      # Main dashboard with sidebar, cards & tables
│   │   ├── data.json               # Earth observation scene records & telemetry
│   │   ├── layout.tsx              # Root dark theme layout
│   │   └── globals.css             # shadcn dark theme tokens (#09090b, #18181b)
│   ├── components/
│   │   ├── app-sidebar.tsx         # Sidebar with brand, quick action & nav
│   │   ├── site-header.tsx         # Top bar with breadcrumb & GitHub link
│   │   ├── section-cards.tsx       # 4 top metric cards with trend badges
│   │   ├── chart-area-interactive.tsx # Time-range interactive area chart
│   │   ├── data-table.tsx          # Earth observation datasets table
│   │   └── satellite-studio.tsx    # Live satellite SR & MC Dropout studio
│   ├── package.json
│   └── tailwind.config.ts
├── server.py                       # High-performance FastAPI backend API
├── run_dev.bat                     # Windows one-click launcher for frontend & backend
├── app.py                          # Standalone Gradio Web Application
├── model.py                        # BhuVistaarNet architecture & Monte Carlo Dropout
├── utils.py                        # Multi-band GeoTIFF loader, metrics, and heatmaps
├── create_samples.py               # Synthetic Sentinel-2 test data generator
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
└── samples/                        # Pre-packaged satellite test scenes
```

---

## 🛠️ Tech Stack

- **Frontend**: Next.js 14 (App Router), React 18, Tailwind CSS, shadcn/ui dark theme, Recharts, Lucide Icons
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Deep Learning Framework**: PyTorch, Torchvision
- **Geospatial & Vision**: Rasterio, OpenCV, Scikit-Image, Pillow
- **Scientific Computing**: NumPy, SciPy, Matplotlib

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
- Python 3.10 to 3.14
- Node.js v18+ (Node v24 supported)
- Git

### 2. Quick Start (Run Both Frontend & Backend)

On Windows, simply run:
```cmd
run_dev.bat
```

Or start the services manually:

**Terminal 1 (FastAPI Backend):**
```bash
python server.py
# API runs on: http://127.0.0.1:8000
```

**Terminal 2 (Next.js shadcn/ui Dashboard):**
```bash
cd frontend
npm run dev
# Dashboard runs on: http://localhost:3000
```

Open your browser at:
👉 **`http://localhost:3000`**

---

## 💻 How to Use the Dashboard

1. **Select or Upload an Image**:
   - Choose one of the built-in SIH 2026 test samples (`Sentinel-2 RGB`, `Sentinel-2 RGB+NIR`, or `Urban Optical`) using the quick-select buttons.
   - Or drag-and-drop your own `.tif` or `.png` satellite file.
2. **Configure Inference**:
   - Adjust the **Monte Carlo Dropout Passes** slider (default: `15`).
   - Select your preferred uncertainty heatmap colormap (e.g., `turbo`).
3. **Generate Super-Resolution**:
   - Click **🚀 Generate Super Resolution**.
4. **Inspect Results**:
   - View the **Side-by-Side Comparison** tab.
   - Check the **Epistemic Uncertainty Heatmap** to identify areas of model variance.
   - Switch to **False-Color NIR Composite** to analyze agricultural vegetation.
   - Inspect the **Geospatial Metadata** card for CRS, dimensions, and band count.
5. **Download Product**:
   - Click the download card to save the georeferenced output for GIS software.

---

## 🏆 Smart India Hackathon 2026 Innovation Highlights

- **Physics-Informed Satellite Handling**: Unlike standard super-resolution networks that only work on 8-bit web images, BhuVistaar handles raw 16-bit satellite bands and multi-spectral sensors.
- **Safety-Critical Uncertainty Quantification**: By providing Bayesian epistemic uncertainty maps, disaster response coordinators can distinguish between high-confidence reconstructed structures and ambiguous features.
- **Ready for GIS Pipelines**: Directly outputs georeferenced `.tif` files compatible with ISRO Bhuvan, QGIS, and Esri ArcGIS.

---

*Developed with pride by **Team The Outliers** for Smart India Hackathon 2026.*
