# =============================================================================
# FIXED DOCKERFILE - Compatible Versions for Torch 2.5.1
# =============================================================================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=1000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# 1. CREATE CONSTRAINT FILE
# We lock Torch/Numpy here. Pip will REJECT any package that tries to install
# incompatible versions (like marker-pdf 1.10.2 trying to pull torch 2.7).
# =============================================================================
RUN echo "numpy>=1.24.0,<2.0.0" > /app/constraints.txt && \
    echo "torch==2.5.1+cu121" >> /app/constraints.txt && \
    echo "torchvision==0.20.1+cu121" >> /app/constraints.txt && \
    echo "torchaudio==2.5.1+cu121" >> /app/constraints.txt

# Apply constraints globally
ENV PIP_CONSTRAINT=/app/constraints.txt

# =============================================================================
# 2. Install PyTorch (CUDA 12.1)
# =============================================================================
RUN pip install --no-cache-dir --retries 10 \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    && rm -rf /root/.cache/pip

# =============================================================================
# 4. Install Marker & Surya (FORCE LATEST)
# We manually install dependencies first to ensure safety.
# Then we force-install the LATEST Surya (0.16.0) using --no-deps.
# This fixes the "KeyError" (by updating code) AND the "Build Error" (by ignoring Torch check).
# =============================================================================
RUN pip install --no-cache-dir \
    "pydantic>=2.0" \
    "pypdfium2" \
    "pillow" \
    "filetype" \
    "markdownify" \
    "python-dotenv" \
    "pdftext" \
    "albumentations" \
    "PyYAML" \
    && pip install --no-cache-dir --no-deps \
    marker-pdf==1.7.1 \
    surya-ocr==0.16.0 \
    && rm -rf /root/.cache/pip
# =============================================================================
# 4. Install Marker & Surya (THE FINAL BRIDGE)
# Marker 1.7.1 is required for the new 2025 models.
# Surya 0.14.1 is the ONLY version that works with Marker 1.7.1 
# AND supports your PyTorch 2.5 setup.
# =============================================================================
RUN pip install --no-cache-dir \
    marker-pdf==1.7.1 \
    surya-ocr==0.14.1 \
    && rm -rf /root/.cache/pip
# =============================================================================
# 5. Install App Dependencies
# =============================================================================
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# =============================================================================
# 6. VERIFICATION
# =============================================================================
RUN python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})'); assert 'cu121' in torch.__version__" && \
    python -c "import marker.models; print('Marker imported successfully')" && \
    echo "=== Build Verified ==="

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]