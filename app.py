"""
BhuVistaar - AI-Powered Super Resolution Mapping

Web application built with Gradio, PyTorch, Rasterio, OpenCV, and Matplotlib.
Supports multi-band satellite images (RGB, RGB+NIR), Monte Carlo Dropout uncertainty
estimation, side-by-side visual comparisons, and georeferenced GeoTIFF exports.
"""

import os
import tempfile
from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np
import gradio as gr

from model import BhuVistaarModelManager
from utils import (
    load_satellite_image,
    image_to_tensor,
    tensor_to_images,
    generate_uncertainty_heatmap,
    calculate_metrics,
    save_geotiff
)

# Initialize Model Manager (supports dynamic band switching)
model_manager = BhuVistaarModelManager()

# Base sample paths
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
SAMPLE_SENTINEL_RGB = os.path.join(SAMPLES_DIR, "sample_sentinel2_rgb.tif")
SAMPLE_SENTINEL_NIR = os.path.join(SAMPLES_DIR, "sample_sentinel2_rgb_nir.tif")
SAMPLE_URBAN_PNG = os.path.join(SAMPLES_DIR, "sample_urban_lr.png")


def create_side_by_side_comparison(
    lr_preview: np.ndarray,
    sr_rgb: np.ndarray
) -> np.ndarray:
    """
    Creates a high-contrast side-by-side annotated comparison between
    the low-resolution input (bicubic upscaled) and the super-resolved output.
    """
    h_sr, w_sr = sr_rgb.shape[:2]
    # Upscale LR to match SR height and width
    lr_matched = cv2.resize(lr_preview, (w_sr, h_sr), interpolation=cv2.INTER_NEAREST)

    # 4-pixel dividing vertical line
    divider = np.full((h_sr, 4, 3), 255, dtype=np.uint8)

    combined = np.hstack([lr_matched, divider, sr_rgb])

    # Clean institutional banner
    banner_height = 36
    banner = np.full((banner_height, combined.shape[1], 3), 245, dtype=np.uint8)

    # Clean dark slate text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(banner, "Low-Resolution Input (1x)", (16, 24), font, 0.55, (40, 50, 60), 1, cv2.LINE_AA)
    cv2.putText(banner, "Super-Resolved Output (2x)", (w_sr + 20, 24), font, 0.55, (20, 90, 50), 1, cv2.LINE_AA)

    final_comparison = np.vstack([banner, combined])
    return final_comparison


def process_satellite_imagery(
    file_obj,
    num_passes: int = 15,
    colormap_name: str = "turbo"
):
    """
    Main processing pipeline for BhuVistaar.
    Handles GeoTIFF / image ingestion, MC Dropout inference,
    metric computation, and georeferenced output generation.
    """
    if file_obj is None:
        raise gr.Error("Please upload a satellite image or select one of the provided samples.")

    # file_obj can be a string path or a Gradio File object with a .name attribute
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    if not os.path.exists(file_path):
        raise gr.Error(f"Selected file does not exist: {file_path}")

    try:
        # 1. Preprocess satellite image
        normalized_data, metadata, lr_preview = load_satellite_image(file_path)

        # 2. Convert to model tensor
        tensor = image_to_tensor(normalized_data, device=model_manager.device)

        # 3. Perform Super-Resolution with Monte Carlo Dropout
        sr_tensor, unc_tensor = model_manager.super_resolve(tensor, num_passes=int(num_passes))

        # 4. Convert model tensors back to viewable images
        sr_rgb, sr_cir = tensor_to_images(sr_tensor)

        # 5. Generate side-by-side comparison
        side_by_side = create_side_by_side_comparison(lr_preview, sr_rgb)

        # 6. Generate Epistemic Uncertainty Heatmap
        uncertainty_heatmap, unc_stats = generate_uncertainty_heatmap(
            unc_tensor, colormap_name=colormap_name
        )

        # 7. Compute scientific metrics
        metrics = calculate_metrics(lr_preview, sr_rgb)

        # 8. Save output GeoTIFF / Image for download
        temp_dir = tempfile.gettempdir()
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_ext = ".tif" if metadata.get("is_geotiff") else ".png"
        out_filename = f"BhuVistaar_SR_2x_{base_name}{out_ext}"
        out_path = os.path.join(temp_dir, out_filename)
        save_geotiff(out_path, sr_tensor, metadata)

        # 9. Format Markdown Metadata & Inspector
        crs_info = metadata.get("crs", "Non-projected / Local coordinate system")
        dtype_info = metadata.get("dtype", "uint8")
        band_count = metadata.get("count", normalized_data.shape[0])
        orig_h, orig_w = metadata.get("height", lr_preview.shape[0]), metadata.get("width", lr_preview.shape[1])
        sr_h, sr_w = sr_rgb.shape[:2]

        metadata_md = f"""
### Satellite Scene Metadata
- **Source File**: `{metadata.get('file_name', 'Unknown')}`
- **Coordinate Reference System (CRS)**: `{crs_info}`
- **Input Dimensions**: `{orig_w} x {orig_h}` pixels
- **Super-Resolved Dimensions (2x)**: `{sr_w} x {sr_h}` pixels
- **Spectral Bands Processed**: `{band_count}` {'(Multi-Band RGB + NIR)' if band_count >= 4 else '(True Color RGB)'}
- **Radiometric Bit Depth**: `{dtype_info}`
- **GeoTIFF Georeferenced**: `{'Yes (Spatial transform preserved)' if metadata.get('is_geotiff') else 'Standard Imagery'}`
"""

        # Format CIR tab visibility/content
        cir_notice = ""
        if sr_cir is None:
            sr_cir = np.copy(sr_rgb)
            cir_notice = "Uploaded image does not contain a 4th Near-Infrared (NIR) band. Displaying standard RGB."
        else:
            cir_notice = "4-band RGB+NIR scene detected. False-Color Infrared composite (NIR, Red, Green) rendered above."

        # Return all UI outputs
        return (
            lr_preview,             # Original Low-Res
            sr_rgb,                  # Super-Resolved Output
            uncertainty_heatmap,     # Uncertainty Heatmap
            side_by_side,            # Side-by-side composite
            sr_cir,                  # False-Color NIR composite
            cir_notice,              # Notice about CIR
            f"{metrics['psnr_db']} dB",
            f"{metrics['ssim']}",
            f"{metrics['sharpness_gain_ratio']}x ({metrics['sharpness_lr']} -> {metrics['sharpness_sr']})",
            f"{unc_stats['mean_uncertainty']:.4f} (sigma max: {unc_stats['max_uncertainty']:.4f})",
            metadata_md,
            out_path                 # Downloadable file
        )

    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")


# Clean Professional Styling
custom_css = """
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: #f8fafc !important;
    color: #1e293b;
}
.header-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.header-tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #475569;
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 3px 8px;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.header-title {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
    margin: 4px 0 2px 0;
}
.header-subtitle {
    font-size: 14px;
    font-weight: 500;
    color: #334155;
    margin-bottom: 8px;
}
.header-desc {
    font-size: 13px;
    color: #64748b;
    margin: 0;
    line-height: 1.5;
}
"""

with gr.Blocks(title="BhuVistaar - Satellite Super Resolution") as demo:
    # Header Section
    gr.HTML("""
    <div class="header-card">
        <span class="header-tag">Satellite Super-Resolution</span>
        <h1 class="header-title">BhuVistaar</h1>
        <div class="header-subtitle">AI-Powered Geospatial Mapping</div>
        <p class="header-desc">
            Geospatial super-resolution prototype for multi-band Earth observation imagery (RGB and NIR).
            Features 2x sub-pixel residual reconstruction with Monte Carlo Dropout epistemic uncertainty quantification.
        </p>
    </div>
    """)

    with gr.Row():
        # Left Column: Ingestion & Controls
        with gr.Column(scale=4):
            gr.Markdown("#### 1. Satellite Image Input")
            input_file = gr.File(
                label="Upload Satellite File (.tif, .tiff, .png, .jpg)",
                file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                file_count="single"
            )

            gr.Markdown("##### Built-in Test Samples:")
            with gr.Row():
                btn_s2_rgb = gr.Button("Sentinel-2 RGB (.tif)", size="sm")
                btn_s2_nir = gr.Button("Sentinel-2 RGB+NIR (.tif)", size="sm")
                btn_urban = gr.Button("Urban Optical (.png)", size="sm")

            gr.Markdown("#### 2. Inference Settings")
            mc_passes_slider = gr.Slider(
                minimum=5,
                maximum=30,
                value=15,
                step=1,
                label="Monte Carlo Dropout Passes (T)",
                info="Number of stochastic forward passes for uncertainty quantification (Default: 15)"
            )

            colormap_dropdown = gr.Dropdown(
                choices=["turbo", "magma", "inferno", "viridis"],
                value="turbo",
                label="Uncertainty Colormap",
                info="Color palette for predictive standard deviation"
            )

            generate_btn = gr.Button(
                "Generate Super Resolution",
                variant="primary",
                size="lg"
            )

            gr.Markdown("#### 3. Export Asset")
            download_file = gr.File(
                label="Download Super-Resolved Product"
            )

        # Right Column: Results Dashboard
        with gr.Column(scale=8):
            gr.Markdown("#### Results & Analysis")

            # Metrics Row
            with gr.Row():
                metric_psnr = gr.Textbox(label="PSNR (vs Bicubic)", value="--", interactive=False)
                metric_ssim = gr.Textbox(label="SSIM Quality Index", value="--", interactive=False)
                metric_sharpness = gr.Textbox(label="Sharpness Enhancement", value="--", interactive=False)
                metric_uncertainty = gr.Textbox(label="Mean Uncertainty (sigma)", value="--", interactive=False)

            # Tabbed View for Inspection
            with gr.Tabs():
                with gr.TabItem("Side-by-Side Comparison"):
                    out_side_by_side = gr.Image(
                        label="Low-Resolution vs Super-Resolved (2x) Direct Comparison",
                        interactive=False
                    )

                with gr.TabItem("Individual Views"):
                    with gr.Row():
                        out_lr = gr.Image(label="Original Low-Resolution Input (1x)", interactive=False)
                        out_sr = gr.Image(label="Super-Resolved Output (2x)", interactive=False)

                with gr.TabItem("Uncertainty Heatmap"):
                    out_uncertainty = gr.Image(
                        label="Epistemic Uncertainty Map (Predictive Standard Deviation)",
                        interactive=False
                    )
                    gr.Markdown("""
                    **Uncertainty Interpretation**: Higher values highlight regions of model variance (e.g. sharp texture transitions and high-frequency boundaries), while lower values indicate high predictive stability.
                    """)

                with gr.TabItem("False-Color NIR Composite"):
                    out_cir_notice = gr.Markdown("Select a 4-band image to inspect the NIR composite.")
                    out_cir = gr.Image(
                        label="False-Color Infrared Composite (NIR - Red - Green)",
                        interactive=False
                    )

                with gr.TabItem("Geospatial Metadata"):
                    out_metadata = gr.Markdown("Metadata will appear after processing an image.")

    # Wiring Sample Buttons
    btn_s2_rgb.click(
        fn=lambda: SAMPLE_SENTINEL_RGB,
        inputs=[],
        outputs=[input_file]
    )
    btn_s2_nir.click(
        fn=lambda: SAMPLE_SENTINEL_NIR,
        inputs=[],
        outputs=[input_file]
    )
    btn_urban.click(
        fn=lambda: SAMPLE_URBAN_PNG,
        inputs=[],
        outputs=[input_file]
    )

    # Wiring Generate Button
    generate_btn.click(
        fn=process_satellite_imagery,
        inputs=[input_file, mc_passes_slider, colormap_dropdown],
        outputs=[
            out_lr,
            out_sr,
            out_uncertainty,
            out_side_by_side,
            out_cir,
            out_cir_notice,
            metric_psnr,
            metric_ssim,
            metric_sharpness,
            metric_uncertainty,
            out_metadata,
            download_file
        ]
    )


if __name__ == "__main__":
    # Launch local server with clean default theme and professional styling
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Default(),
        css=custom_css
    )
