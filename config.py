"""
BhuVistaar - Dynamic Configuration Manager
Loads environment variables from .env with full type-casting, sensible defaults,
and path resolution across the entire system.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Automatically locate and load .env file from project root
ENV_FILE = BASE_DIR / ".env"
if HAS_DOTENV and ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)
elif HAS_DOTENV:
    # Also attempt standard dotenv discovery
    load_dotenv()


def _get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on", "t")


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _get_list(key: str, default: Optional[List[str]] = None) -> List[str]:
    val = os.getenv(key)
    if val is None or not val.strip():
        return default or ["*"]
    items = [item.strip() for item in val.split(",") if item.strip()]
    return items if items else (default or ["*"])


def _resolve_path(raw_path: str, default: str) -> Path:
    val = raw_path if raw_path and raw_path.strip() else default
    p = Path(val.strip())
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    return p


class Settings:
    """Dynamic configuration settings loaded from environment & .env file."""

    def __init__(self):
        self.reload(override=False)

    def reload(self, override: bool = False):
        """Reload configuration from active environment."""
        if HAS_DOTENV and ENV_FILE.exists():
            load_dotenv(dotenv_path=ENV_FILE, override=override)

        self.BASE_DIR: Path = BASE_DIR

        # --------------------------------------------------------------------
        # Server Settings
        # --------------------------------------------------------------------
        self.HOST: str = os.getenv("HOST", "127.0.0.1")
        self.PORT: int = _get_int("PORT", 8000)
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
        self.DEBUG: bool = _get_bool("DEBUG", default=(self.ENVIRONMENT == "development"))
        self.CORS_ORIGINS: List[str] = _get_list(
            "CORS_ORIGINS",
            default=["http://localhost:3000", "http://127.0.0.1:3000", "*"]
        )

        # --------------------------------------------------------------------
        # Storage & Cache Directories
        # --------------------------------------------------------------------
        self.DATA_CACHE_DIR: Path = _resolve_path(os.getenv("DATA_CACHE_DIR", ""), "data/cache")
        custom_temp = os.getenv("TEMP_OUTPUT_DIR", "")
        if custom_temp:
            self.TEMP_OUTPUT_DIR: Path = _resolve_path(custom_temp, "data/cache/outputs")
        else:
            self.TEMP_OUTPUT_DIR: Path = (self.DATA_CACHE_DIR / "outputs").resolve()

        self.SAMPLES_DIR: Path = _resolve_path(os.getenv("SAMPLES_DIR", ""), "samples")

        # Ensure essential directories exist
        self.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------------------
        # Deep Learning & Inference Settings
        # --------------------------------------------------------------------
        self.MODEL_CHECKPOINT_PATH: Path = _resolve_path(
            os.getenv("MODEL_CHECKPOINT_PATH", ""),
            "weights/RealESRGAN_x4plus.pth"
        )
        self.DEVICE: str = os.getenv("DEVICE", "auto").strip().lower()
        self.SCALE_FACTOR: int = _get_int("SCALE_FACTOR", 4)
        self.DEFAULT_MC_PASSES: int = _get_int("DEFAULT_MC_PASSES", 15)
        self.DEFAULT_COLORMAP: str = os.getenv("DEFAULT_COLORMAP", "turbo").strip().lower()

        # --------------------------------------------------------------------
        # Copernicus Data Space Ecosystem (CDSE) API
        # --------------------------------------------------------------------
        self.COPERNICUS_CLIENT_ID: str = os.getenv("COPERNICUS_CLIENT_ID", "").strip()
        self.COPERNICUS_CLIENT_SECRET: str = os.getenv("COPERNICUS_CLIENT_SECRET", "").strip()
        self.COPERNICUS_AUTH_URL: str = os.getenv(
            "COPERNICUS_AUTH_URL",
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        ).strip()
        self.COPERNICUS_STAC_URL: str = os.getenv(
            "COPERNICUS_STAC_URL",
            "https://stac.dataspace.copernicus.eu/v1/search"
        ).strip()

        # --------------------------------------------------------------------
        # ISRO Bhoonidhi STAC API
        # --------------------------------------------------------------------
        self.BHOONIDHI_USERNAME: str = os.getenv("BHOONIDHI_USERNAME", "").strip()
        self.BHOONIDHI_PASSWORD: str = os.getenv("BHOONIDHI_PASSWORD", "").strip()
        self.BHOONIDHI_API_URL: str = os.getenv(
            "BHOONIDHI_API_URL",
            "https://bhoonidhi.nrsc.gov.in/bhoonidhi-api/stac/search"
        ).strip()

        # --------------------------------------------------------------------
        # Frontend URL Configuration
        # --------------------------------------------------------------------
        self.FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000").strip()
        self.BACKEND_URL: str = os.getenv(
            "BACKEND_URL",
            f"http://{self.HOST}:{self.PORT}"
        ).strip()

    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """Returns safe representation of settings (masking secrets)."""
        def mask(s: str) -> str:
            if not s:
                return "<not set>"
            if len(s) <= 4:
                return "****"
            return s[:2] + "****" + s[-2:]

        return {
            "HOST": self.HOST,
            "PORT": self.PORT,
            "ENVIRONMENT": self.ENVIRONMENT,
            "DEBUG": self.DEBUG,
            "CORS_ORIGINS": self.CORS_ORIGINS,
            "DATA_CACHE_DIR": str(self.DATA_CACHE_DIR),
            "TEMP_OUTPUT_DIR": str(self.TEMP_OUTPUT_DIR),
            "SAMPLES_DIR": str(self.SAMPLES_DIR),
            "MODEL_CHECKPOINT_PATH": str(self.MODEL_CHECKPOINT_PATH),
            "DEVICE": self.DEVICE,
            "SCALE_FACTOR": self.SCALE_FACTOR,
            "DEFAULT_MC_PASSES": self.DEFAULT_MC_PASSES,
            "DEFAULT_COLORMAP": self.DEFAULT_COLORMAP,
            "COPERNICUS_CLIENT_ID": mask(self.COPERNICUS_CLIENT_ID) if mask_secrets else self.COPERNICUS_CLIENT_ID,
            "COPERNICUS_CLIENT_SECRET": "******" if (mask_secrets and self.COPERNICUS_CLIENT_SECRET) else self.COPERNICUS_CLIENT_SECRET,
            "COPERNICUS_AUTH_URL": self.COPERNICUS_AUTH_URL,
            "COPERNICUS_STAC_URL": self.COPERNICUS_STAC_URL,
            "BHOONIDHI_USERNAME": mask(self.BHOONIDHI_USERNAME) if mask_secrets else self.BHOONIDHI_USERNAME,
            "BHOONIDHI_PASSWORD": "******" if (mask_secrets and self.BHOONIDHI_PASSWORD) else self.BHOONIDHI_PASSWORD,
            "BHOONIDHI_API_URL": self.BHOONIDHI_API_URL,
            "FRONTEND_URL": self.FRONTEND_URL,
            "BACKEND_URL": self.BACKEND_URL,
        }


# Singleton configuration instance
settings = Settings()
