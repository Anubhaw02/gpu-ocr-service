FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# 3. Install PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir --retries 10 \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# 4. Install The Stability Path Stack
RUN pip install --no-cache-dir \
    "pydantic==2.12.5" \
    "pypdfium2" \
    "pillow" \
    "filetype" \
    "markdownify" \
    "python-dotenv" \
    && pip install --no-cache-dir \
    marker-pdf==1.2.1 \
    surya-ocr==0.8.3 \
    transformers==4.45.2 \
    tokenizers==0.20.3 \
    regex==2024.4.28 \
    sympy==1.13.1 \
    click==8.1.7



# 5. Install Remaining Requirements
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt \
    && pip install --no-cache-dir h11==0.14.0 starlette==0.36.3 click==8.1.7



# 6. Copy Code
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]