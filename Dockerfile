# ==============================================================================
# BhuVistaar V2 - Production Multi-Stage Dockerfile for Render & Cloud
# Stage 1: Build Next.js Static Export
# Stage 2: Production Python & Fullstack FastAPI Server
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Frontend Builder Stage
# ------------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ ./
ENV NEXT_OUTPUT_EXPORT=true
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build

# ------------------------------------------------------------------------------
# 2. Production Runner Stage
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    HOST=0.0.0.0 \
    ENVIRONMENT=production \
    DEVICE=cpu

WORKDIR /app

# Install essential system libraries for OpenCV, image processing & Rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements (using CPU wheels for faster lightweight builds)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Create non-root user (UID 1000 is required by Hugging Face Spaces)
RUN useradd -m -u 1000 user && \
    mkdir -p /app/data/cache/outputs /app/weights && \
    chown -R user:user /app

# Copy backend source code & assets
COPY --chown=user:user config.py ./
COPY --chown=user:user app/ ./app/
COPY --chown=user:user training/ ./training/
COPY --chown=user:user weights/ ./weights/
COPY --chown=user:user samples/ ./samples/

# Download pretrained RealESRGAN weights if not present in git checkout
RUN if [ ! -s weights/RealESRGAN_x4plus.pth ]; then \
        echo "Downloading RealESRGAN_x4plus.pth backbone weights..." && \
        curl -L -f -o weights/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth ; \
    fi && \
    chown -R user:user /app/weights && \
    chmod -R 777 /app/data /app/weights

# Copy built frontend static export from Stage 1
COPY --from=frontend-builder --chown=user:user /app/frontend/out ./frontend/out

# Switch to non-root user
USER user

EXPOSE 7860
EXPOSE 8000

# Hugging Face Spaces uses PORT 7860; Render passes dynamic $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
