"""
Configuration for AI models.
Add your quantized models to the 'models' directory and configure them here.
"""

import os
import glob

# Base directory for models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Upload directory for images
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')

def find_gguf_models():
    """Find all GGUF model files in the models directory."""
    gguf_files = glob.glob(os.path.join(MODELS_DIR, '*.gguf'))
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
    return models

# Available models configuration
# Each model has: name, type (text-to-image or image-to-image), path
AVAILABLE_MODELS = {
    "qwen-image": {
        "name": "Qwen Image Edit (GGUF)",
        "type": "image-to-image",
        "path": os.path.join(MODELS_DIR, "qwen-image-edit-2511-Q5_K_M.gguf"),
        "description": "Qwen model for image editing with prompt support (quantized GGUF)",
        "supports_multiple_images": True,
        "format": "gguf"
    }
}

# Auto-discover GGUF models and add them to available models
AVAILABLE_MODELS.update(find_gguf_models())

# Server configuration
HOST = "0.0.0.0"
PORT = 5005
DEBUG = True

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
