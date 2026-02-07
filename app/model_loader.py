import torch
import logging
from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter

# 1. Setup Logging
# We use logging instead of 'print' because in Docker/Cloud, 
# print statements can get lost or buffered.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Global Variable (The Singleton)
# This starts as None. Once loaded, it holds the heavy model.
_model_instance = None

def load_model_to_memory():
    """
    Loads the Marker model into GPU memory.
    This function is designed to be called ONLY ONCE at startup.
    """
    global _model_instance
    
    # If model is already loaded, don't reload it!
    if _model_instance is not None:
        logger.info("Model is already loaded.")
        return _model_instance

    logger.info("⏳ Loading Marker Model... This may take time.")
    
    try:
        # 3. GPU Detection
        # This line is critical for your GTX 1650 Ti.
        # It asks PyTorch: "Do I have a CUDA-capable GPU?"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Device Detected: {device.upper()}")

        # 4. Create the Model
        # We explicitly pass the device so it loads into VRAM, not RAM.
        model_dict = create_model_dict(device=device)
        _model_instance = PdfConverter(artifact_dict=model_dict)
        
        logger.info("✅ Model successfully loaded into memory!")
        return _model_instance

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise e

def get_model():
    """
    Returns the loaded model instance.
    If it wasn't loaded for some reason, it attempts to load it.
    """
    if _model_instance is None:
        return load_model_to_memory()
    return _model_instance