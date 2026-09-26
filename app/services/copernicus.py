"""
Copernicus Data Space Ecosystem (CDSE) STAC Client
Fetches Sentinel-2 L2A BOA reflectance tiles (B04, B03, B02, B08)
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
import requests

try:
    from config import settings
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import settings

logger = logging.getLogger("bhuvistaar.copernicus")

COPERNICUS_AUTH_URL = settings.COPERNICUS_AUTH_URL
COPERNICUS_STAC_URL = settings.COPERNICUS_STAC_URL
CACHE_DIR = str(settings.DATA_CACHE_DIR / "sentinel2")

# Token cache
_token_cache = {
    "access_token": None,
    "expires_at": 0
}

def load_env_credentials():
    """Load credentials dynamically from configuration."""
    return settings.COPERNICUS_CLIENT_ID, settings.COPERNICUS_CLIENT_SECRET

def get_copernicus_token() -> Optional[str]:
    """Retrieve and cache OAuth2 access token for Copernicus Data Space."""
    client_id, client_secret = load_env_credentials()
    if not client_id or not client_secret or client_id.startswith("your_"):
        return None

    now = time.time()
    if _token_cache["access_token"] and _token_cache["expires_at"] > now + 60:
        return _token_cache["access_token"]

    try:
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials"
        }
        resp = requests.post(COPERNICUS_AUTH_URL, data=data, timeout=15)
        if resp.status_code == 200:
            token_json = resp.json()
            _token_cache["access_token"] = token_json.get("access_token")
            expires_in = token_json.get("expires_in", 3600)
            _token_cache["expires_at"] = now + expires_in
            return _token_cache["access_token"]
        else:
            logger.warning(f"Failed to fetch Copernicus token: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        logger.warning(f"Error fetching Copernicus token: {e}")
        return None

def search_sentinel2_scenes(
    bbox: List[float],
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    max_cloud_cover: float = 20.0,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search Sentinel-2 L2A scenes via Copernicus STAC API.
    bbox: [min_lon, min_lat, max_lon, max_lat]
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    payload = {
        "collections": ["SENTINEL-2"],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "query": {
            "cloudCover": {
                "lte": max_cloud_cover
            }
        },
        "limit": limit
    }
    
    token = get_copernicus_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    try:
        resp = requests.post(COPERNICUS_STAC_URL, json=payload, headers=headers, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            results = []
            for feat in features:
                props = feat.get("properties", {})
                scene_info = {
                    "id": feat.get("id"),
                    "collection": feat.get("collection", "SENTINEL-2"),
                    "datetime": props.get("datetime"),
                    "cloud_cover": props.get("cloudCover", props.get("eo:cloud_cover", 0.0)),
                    "bbox": feat.get("bbox", bbox),
                    "geometry": feat.get("geometry"),
                    "assets": {
                        band: asset.get("href")
                        for band, asset in feat.get("assets", {}).items()
                        if band in ["B04", "B03", "B02", "B08", "visual", "red", "green", "blue", "nir"]
                    }
                }
                results.append(scene_info)
            return results
    except Exception as e:
        logger.warning(f"STAC search request failed: {e}")

    # Fallback to local sample scenes if remote query unavailable or offline
    return get_cached_or_sample_scenes(bbox)

def get_cached_or_sample_scenes(bbox: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """Provide realistic sample Sentinel-2 scenes for demonstration/offline use."""
    return [
        {
            "id": "S2A_MSIL2A_20240415T052651_N0510_R105_T43REQ_20240415T083412",
            "collection": "SENTINEL-2",
            "datetime": "2024-04-15T05:26:51Z",
            "cloud_cover": 2.4,
            "bbox": bbox or [77.10, 28.55, 77.30, 28.75],
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [77.10, 28.55],
                    [77.30, 28.55],
                    [77.30, 28.75],
                    [77.10, 28.75],
                    [77.10, 28.55]
                ]]
            },
            "assets": {
                "B04": "local://samples/S2A_T43REQ_B04.tif",
                "B03": "local://samples/S2A_T43REQ_B03.tif",
                "B02": "local://samples/S2A_T43REQ_B02.tif",
                "B08": "local://samples/S2A_T43REQ_B08.tif"
            },
            "is_sample": True,
            "location_name": "New Delhi / NCR Region"
        },
        {
            "id": "S2B_MSIL2A_20240320T045709_N0510_R033_T44QND_20240320T074522",
            "collection": "SENTINEL-2",
            "datetime": "2024-03-20T04:57:09Z",
            "cloud_cover": 0.8,
            "bbox": bbox or [80.15, 12.95, 80.35, 13.15],
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [80.15, 12.95],
                    [80.35, 12.95],
                    [80.35, 13.15],
                    [80.15, 13.15],
                    [80.15, 12.95]
                ]]
            },
            "assets": {
                "B04": "local://samples/S2B_T44QND_B04.tif",
                "B03": "local://samples/S2B_T44QND_B03.tif",
                "B02": "local://samples/S2B_T44QND_B02.tif",
                "B08": "local://samples/S2B_T44QND_B08.tif"
            },
            "is_sample": True,
            "location_name": "Chennai Coastal / Urban Corridor"
        }
    ]
