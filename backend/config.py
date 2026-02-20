"""
Configuration for AI models.
Add your quantized models to the 'models' directory and configure them here.
"""

import os

# Base directory for models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Upload directory for images
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')

# Available models configuration
# Each model has: name, type (text-to-image or image-to-image), path
AVAILABLE_MODELS = {
    "qwen-image": {
        "name": "Qwen Image (Image-to-Image)",
        "type": "image-to-image",
        "path": os.path.join(MODELS_DIR, "qwen-image"),
        "description": "Qwen model for image editing with prompt support",
        "supports_multiple_images": True
    },
    "sd-xl": {
        "name": "Stable Diffusion XL (Text-to-Image)",
        "type": "text-to-image",
        "path": os.path.join(MODELS_DIR, "sdxl"),
        "description": "Stable Diffusion XL for text-to-image generation",
        "supports_multiple_images": False
    },
    "sd-turbo": {
        "name": "SD Turbo (Text-to-Image Fast)",
        "type": "text-to-image",
        "path": os.path.join(MODELS_DIR, "sdturbo"),
        "description": "Stable Diffusion Turbo for fast text-to-image generation",
        "supports_multiple_images": False
    }
}

# Server configuration
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
