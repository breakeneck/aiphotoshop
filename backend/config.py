"""
Configuration for AI models.
Add your quantized models to the 'models' directory and configure them here.
"""

import os
import glob
import logging

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for models - use absolute path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Upload directory for images
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

# Debug output - use print to ensure it shows before logging is fully configured
print(f"[CONFIG] BASE_DIR: {BASE_DIR}")
print(f"[CONFIG] MODELS_DIR: {MODELS_DIR}")
print(f"[CONFIG] MODELS_DIR exists: {os.path.exists(MODELS_DIR)}")

def find_gguf_models():
    """Find all GGUF model files in the models directory."""
    search_pattern = os.path.join(MODELS_DIR, '*.gguf')
    print(f"[CONFIG] Searching for GGUF files with pattern: {search_pattern}")
    
    gguf_files = glob.glob(search_pattern)
    print(f"[CONFIG] Found GGUF files: {gguf_files}")
    
    models = {}
    for filepath in gguf_files:
        filename = os.path.basename(filepath)
        model_id = filename.replace('.gguf', '').replace('-', '_').lower()
        # Simplified model ID for cleaner display
        model_id = model_id.replace('_q5_k_m', '').replace('_q4_k_m', '').replace('_q8_0', '')
        
        models[model_id] = {
            "name": filename.replace('.gguf', ''),
            "type": "image-to-image",
            "path": filepath,
            "description": f"GGUF quantized model: {filename}",
            "supports_multiple_images": True,
            "format": "gguf"
        }
        print(f"[CONFIG] Added model: {model_id} from {filepath} (format: {models[model_id]['format']})")
    return models

# Available models configuration
# Start with auto-discovered GGUF models
AVAILABLE_MODELS = find_gguf_models()

# Also check for specific model file if auto-discovery didn't find it
specific_model_path = os.path.join(MODELS_DIR, "qwen-image-edit-2511-Q5_K_M.gguf")
print(f"[CONFIG] Checking specific model path: {specific_model_path}")
print(f"[CONFIG] Specific model exists: {os.path.exists(specific_model_path)}")

if os.path.exists(specific_model_path) and "qwen_image_edit_2511" not in AVAILABLE_MODELS:
    AVAILABLE_MODELS["qwen-image"] = {
        "name": "Qwen Image Edit (GGUF)",
        "type": "image-to-image",
        "path": specific_model_path,
        "description": "Qwen model for image editing with prompt support (quantized GGUF)",
        "supports_multiple_images": True,
        "format": "gguf"
    }
    print(f"[CONFIG] Added specific model: qwen-image (format: gguf)")

# Log discovered models
print(f"[CONFIG] Available models: {list(AVAILABLE_MODELS.keys())}")

# Server configuration
HOST = "0.0.0.0"
PORT = 5005
DEBUG = True

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
