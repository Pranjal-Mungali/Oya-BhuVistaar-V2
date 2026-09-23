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
    PORT=8000 \
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

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code & assets
COPY config.py model.py server.py utils.py ./
COPY backend/ ./backend/
COPY weights/ ./weights/
COPY models/ ./models/
COPY samples/ ./samples/
COPY preprocessing/ ./preprocessing/

# Download pretrained RealESRGAN weights if not present in git checkout
RUN mkdir -p weights && \
    if [ ! -s weights/RealESRGAN_x4plus.pth ]; then \
        echo "Downloading RealESRGAN_x4plus.pth backbone weights..." && \
        curl -L -f -o weights/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth ; \
    fi

# Copy built frontend static export from Stage 1
COPY --from=frontend-builder /app/frontend/out ./frontend/out

# Ensure cache directories exist
RUN mkdir -p data/cache/outputs

EXPOSE 8000

# Render dynamically sets $PORT environment variable.
# Start fullstack FastAPI server binding to 0.0.0.0:$PORT
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
