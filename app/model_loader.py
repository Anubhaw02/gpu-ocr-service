import torch
import logging
from safetensors.torch import load_file

# Existing Marker imports
from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Global Variables (Singletons)
_marker_instance = None
_surya_instance = None

def load_model_to_memory():
    """
    Loads the Marker model into GPU memory.
    """
    global _marker_instance
    if _marker_instance is not None:
        logger.info("Marker model is already loaded.")
        return _marker_instance

    logger.info("⏳ Loading Marker Model... This may take time.")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Device Detected: {device.upper()}")

        model_dict = create_model_dict(device=device)
        _marker_instance = PdfConverter(artifact_dict=model_dict)

        logger.info("✅ Marker Model successfully loaded into memory!")
        return _marker_instance
    except Exception as e:
        logger.error(f"❌ Failed to load Marker model: {e}")
        raise e

def get_model():
    """
    Returns the loaded Marker model instance.
    """
    if _marker_instance is None:
        return load_model_to_memory()
    return _marker_instance

# --- NEW: Surya loader ---
def load_surya_layout_model():
    """
    Loads Surya's layout model from local safetensors.
    """
    global _surya_instance
    if _surya_instance is not None:
        logger.info("Surya Layout model is already loaded.")
        return _surya_instance

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Device Detected: {device.upper()}")

        state_dict = load_file("surya_layout/model.safetensors")

        # Use Surya’s actual architecture with config
        from surya.model.layout.model import SuryaLayoutModel, SuryaLayoutConfig
        config = SuryaLayoutConfig()  # instantiate config
        model = SuryaLayoutModel(config)  # pass config into model

        model.load_state_dict(state_dict, strict=False)  # strict=False helps with minor mismatches

        _surya_instance = model.to(device)
        logger.info("✅ Surya Layout Model successfully loaded into memory!")
        return _surya_instance
    except Exception as e:
        logger.error(f"❌ Failed to load Surya Layout model: {e}")
        raise e
