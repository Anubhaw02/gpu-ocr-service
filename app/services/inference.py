import os
import logging

# Setup logging
logger = logging.getLogger(__name__)

def run_inference(model, pdf_path: str):
    """
    Takes a loaded Marker model and a path to a PDF file.
    Returns the full markdown string.
    """
    
    # 1. Validation checks
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file does not exist at path: {pdf_path}")

    logger.info(f"📄 Starting Inference on: {pdf_path}")

    # 2. MOCK MODE (Since you don't have the model yet)
    # This lets you test the API flow without the heavy AI model.
    if model is None:
        logger.warning("⚠️ Model is NOT loaded. Returning dummy data for testing.")
        return "# Dummy Markdown\n\nThis is a placeholder response because the model is missing."

    # 3. Real Inference (The "Heavy Lifting")
    try:
        # The Marker library uses .convert_single_pdf() or similar methods
        # We wrap it in a try/except block to catch GPU OOM errors
        
        full_text, images, out_meta = model.convert_single_pdf(pdf_path)
        
        logger.info("✅ Inference completed successfully.")
        return full_text

    except Exception as e:
        logger.error(f"❌ Inference Failed: {str(e)}")
        
        # Pro Tip: If GPU runs out of memory, we want to know specifically
        if "CUDA out of memory" in str(e):
            raise RuntimeError("GPU Out of Memory! The PDF is likely too large or complex.")
        
        raise e