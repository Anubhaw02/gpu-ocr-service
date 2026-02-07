# 🚀 GPU-Accelerated OCR Microservice

A production-ready **Document Intelligence API** built for high-performance enterprise pipelines. It converts complex PDFs into clean, structured Markdown using **FastAPI** and **Marker**, fully optimized for **NVIDIA CUDA** acceleration.

## 🏗 System Architecture

This microservice is designed to serve as the ingestion layer for RAG (Retrieval-Augmented Generation) systems.

* **Core Engine:** [Marker](https://github.com/VikParuchuri/marker) (PyTorch) for layout-aware PDF extraction.
* **API Framework:** **FastAPI** (Async/Non-blocking).
* **Compute Strategy:** **GPU-First** with automatic CUDA detection.
    * *If GPU found:* Loads FP16 precision models into VRAM.
    * *If CPU only:* Fallback mode (with warning logs).
* **Dependency Strategy:** Implements a **Force-Override Build** to combine the latest 2025 AI models (Surya 0.16.0) with stable Enterprise PyTorch drivers (2.5.1), bypassing strict upstream version checks.

---

## ✨ Key Features

* **⚡ Zero-Latency Inference:** Uses a global singleton pattern (`model_loader.py`) to keep models resident in GPU memory.
* **🔍 Deep Observability:** Custom `/health/detailed` endpoint exposes real-time CUDA tensor core availability and dependency versions.
* **🛡️ Robust Validation:** Enforces strict file type checks (`.pdf` only) and manages temporary file cleanup to prevent disk bloat.
* **🧹 Automatic Resource Management:** Clears GPU cache (`torch.cuda.empty_cache`) during shutdown events.

---

## 🚀 Getting Started

### Prerequisites

* **Docker Desktop** (WSL2 Backend on Windows)
* **NVIDIA Drivers** (Game Ready or Studio drivers)
* **NVIDIA Container Toolkit**

### 1. Build the Container
The build uses a custom `--no-deps` strategy to force-install the latest OCR engines on stable CUDA drivers.
```bash
docker build -t gpu-ocr-service .