"""
Generate sample satellite images for BhuVistaar prototype testing:
1. sample_sentinel2_rgb.tif (3-band GeoTIFF, uint16 Sentinel-2 simulation)
2. sample_sentinel2_rgb_nir.tif (4-band RGB+NIR GeoTIFF, uint16 Sentinel-2 L2A simulation)
3. sample_urban_lr.png (3-band standard RGB satellite patch)
"""

import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

def generate_samples():
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(samples_dir, exist_ok=True)

    h, w = 128, 128
    y, x = np.mgrid[0:h, 0:w]

    # 1. Base terrain features
    # River winding through the terrain
    river_path = 64 + 20 * np.sin(x / 15.0)
    river_mask = (np.abs(y - river_path) < 7)

    # Agricultural plots (grid variation)
    field_grid = ((x // 24) % 2 == 0) ^ ((y // 24) % 2 == 0)

    # Forested dense vegetation patch (top-right quadrant)
    forest_mask = ((x > 70) & (y < 60)).astype(np.float32)

    # Roads / Urban grid lines
    roads_mask = ((x % 32 == 0) | (y % 32 == 0))

    # Construct Sentinel-2 bands (16-bit scaled reflectances [0, 10000])
    # Blue band (B2)
    b_blue = np.full((h, w), 800, dtype=np.float32)
    b_blue[river_mask] = 1200
    b_blue[field_grid] += 300
    b_blue[roads_mask] = 2200

    # Green band (B3)
    b_green = np.full((h, w), 1000, dtype=np.float32)
    b_green[river_mask] = 800
    b_green[forest_mask > 0] = 1400
    b_green[field_grid] += 600
    b_green[roads_mask] = 2300

    # Red band (B4)
    b_red = np.full((h, w), 900, dtype=np.float32)
    b_red[river_mask] = 400
    b_red[forest_mask > 0] = 500  # Chlorophyll absorption
    b_red[field_grid] += 400
    b_red[roads_mask] = 2400

    # Near Infrared band (B8 - NIR)
    b_nir = np.full((h, w), 1800, dtype=np.float32)
    b_nir[river_mask] = 150  # Strong water absorption
    b_nir[forest_mask > 0] = 5800  # Massive leaf reflectance
    b_nir[field_grid] += 2200
    b_nir[roads_mask] = 2500

    # Add subtle natural texture noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 50, (h, w))
    b_blue = np.clip(b_blue + noise, 100, 9500).astype(np.uint16)
    b_green = np.clip(b_green + noise, 100, 9500).astype(np.uint16)
    b_red = np.clip(b_red + noise, 100, 9500).astype(np.uint16)
    b_nir = np.clip(b_nir + noise, 100, 9500).astype(np.uint16)

    # GeoTIFF Spatial parameters: Delhi NCR coordinates for SIH
    west, south, east, north = 77.10, 28.50, 77.25, 28.65
    transform = from_bounds(west, south, east, north, w, h)
    crs = CRS.from_epsg(4326)

    # File 1: 3-band RGB GeoTIFF
    rgb_path = os.path.join(samples_dir, "sample_sentinel2_rgb.tif")
    with rasterio.open(
        rgb_path,
        'w',
        driver='GTiff',
        height=h,
        width=w,
        count=3,
        dtype='uint16',
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(b_red, 1)    # Band 1: Red
        dst.write(b_green, 2)  # Band 2: Green
        dst.write(b_blue, 3)   # Band 3: Blue
    print(f"Created: {rgb_path}")

    # File 2: 4-band RGB + NIR GeoTIFF
    rgb_nir_path = os.path.join(samples_dir, "sample_sentinel2_rgb_nir.tif")
    with rasterio.open(
        rgb_nir_path,
        'w',
        driver='GTiff',
        height=h,
        width=w,
        count=4,
        dtype='uint16',
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(b_red, 1)    # Band 1: Red
        dst.write(b_green, 2)  # Band 2: Green
        dst.write(b_blue, 3)   # Band 3: Blue
        dst.write(b_nir, 4)    # Band 4: NIR
    print(f"Created: {rgb_nir_path}")

    # File 3: Standard 8-bit PNG satellite patch
    # Stretch to 0-255 uint8
    r_u8 = ((b_red - b_red.min()) / (b_red.max() - b_red.min()) * 255).astype(np.uint8)
    g_u8 = ((b_green - b_green.min()) / (b_green.max() - b_green.min()) * 255).astype(np.uint8)
    b_u8 = ((b_blue - b_blue.min()) / (b_blue.max() - b_blue.min()) * 255).astype(np.uint8)
    rgb_u8 = np.stack([r_u8, g_u8, b_u8], axis=-1)
    png_path = os.path.join(samples_dir, "sample_urban_lr.png")
    Image.fromarray(rgb_u8).save(png_path)
    print(f"Created: {png_path}")

if __name__ == "__main__":
    generate_samples()
