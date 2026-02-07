import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Import our custom modules
from app.model_loader import load_model_to_memory, get_model
from app.services.inference import run_inference

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the Lifespan (Startup & Shutdown Logic)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup (model loading) and shutdown (cleanup).
    """
    # --- Startup Logic ---
    logger.info("🚀 API is starting up...")
    try:
        load_model_to_memory()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Critical Error: Model failed to load: {e}")
        logger.warning("⚠️  API will start but inference endpoints may fail")
    
    yield
    
    # --- Shutdown Logic ---
    logger.info("🛑 API is shutting down...")
    # Optional: Clear GPU cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✅ GPU cache cleared")
    except Exception as e:
        logger.error(f"❌ Error during shutdown cleanup: {e}")

# Initialize the App
app = FastAPI(
    title="GPU OCR Service",
    description="Production-ready API for converting PDFs to Markdown using Marker.",
    version="1.0.0",
    lifespan=lifespan
)

# --- HEALTH ENDPOINTS ---

@app.get("/health")
async def health_check():
    """
    Basic health check for Docker HEALTHCHECK and Azure/AWS health probes.
    Returns 200 OK if the service is running.
    """
    model = get_model()
    status = "ready" if model is not None else "model_not_loaded"
    
    return {
        "status": status,
        "service": "gpu-ocr-service"
    }

@app.get("/health/detailed")
async def detailed_health():
    """
    Detailed health check with dependency verification.
    Useful for debugging container runtime and checking GPU availability.
    """
    try:
        import torch
        import numpy as np
        import cv2
        import transformers
        
        model = get_model()
        
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "dependencies": {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "transformers": transformers.__version__
            }
        }
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.get("/")
async def root():
    """
    Root endpoint - redirects to API documentation.
    """
    return {
        "message": "GPU OCR Service API",
        "docs": "/docs",
        "health": "/health",
        "detailed_health": "/health/detailed"
    }

# --- OCR ENDPOINTS ---

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Main OCR endpoint.
    Accepts a PDF file and returns extracted text/markdown.
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Check if model is loaded
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is still initializing."
        )
    
    try:
        # Run inference
        result = await run_inference(file)
        return result
    except Exception as e:
        logger.error(f"Error during OCR inference: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )

# Optional: Add error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )