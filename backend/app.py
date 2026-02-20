"""
Flask Backend API for AI Image Editor.
Provides endpoints for image generation and editing using AI models.
"""

import os
import io
import base64
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

from .config import AVAILABLE_MODELS, UPLOAD_DIR, ALLOWED_EXTENSIONS, allowed_file, HOST, PORT, DEBUG
from .model_manager import model_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder=None, static_url_path=None)
CORS(app)  # Enable CORS for frontend communication

# Log paths for debugging
logger.info(f"__file__: {__file__}")
logger.info(f"Project root: {os.path.dirname(os.path.dirname(__file__))}")
logger.info(f"Static path: {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')}")
logger.info(f"Static path exists: {os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'))}")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure max content length (50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    logger.info(f"API /api/models called. AVAILABLE_MODELS: {list(AVAILABLE_MODELS.keys())}")
    models = []
    for model_id, config in AVAILABLE_MODELS.items():
        logger.info(f"Adding model: {model_id} -> {config.get('name')}")
        models.append({
            "id": model_id,
            "name": config.get("name", model_id),
            "type": config.get("type"),
            "description": config.get("description", ""),
            "supports_multiple_images": config.get("supports_multiple_images", False)
        })
    logger.info(f"Returning {len(models)} models: {models}")
    return jsonify({"models": models})


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current model status."""
    return jsonify(model_manager.get_status())


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load a specific model."""
    data = request.get_json()
    model_id = data.get('model_id')
    
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    
    if model_id not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model: {model_id}"}), 400
    
    try:
        model_config = AVAILABLE_MODELS[model_id]
        model_manager.load_model(model_id, model_config)
        return jsonify({
            "success": True,
            "message": f"Model {model_id} loaded successfully",
            "status": model_manager.get_status()
        })
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/unload_model', methods=['POST'])
def unload_model():
    """Unload current model."""
    try:
        model_manager.unload_current_model()
        return jsonify({
            "success": True,
            "message": "Model unloaded successfully",
            "status": model_manager.get_status()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Generate or edit an image.
    
    For text-to-image:
        - model_id: str
        - prompt: str
        - width: int (optional)
        - height: int (optional)
        - steps: int (optional)
    
    For image-to-image:
        - model_id: str
        - prompt: str
        - images: list of base64 encoded images OR file uploads
    """
    try:
        # Check if it's a file upload or JSON
        if request.content_type and 'multipart/form-data' in request.content_type:
            return handle_multipart_generation()
        else:
            return handle_json_generation()
            
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


def handle_multipart_generation():
    """Handle generation with file uploads."""
    model_id = request.form.get('model_id')
    prompt = request.form.get('prompt', '')
    
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    
    if model_id not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model: {model_id}"}), 400
    
    # Load model if not already loaded
    model_config = AVAILABLE_MODELS[model_id]
    if model_manager.current_model_id != model_id:
        model_manager.load_model(model_id, model_config)
    
    # Get uploaded files
    images = []
    if 'images' in request.files:
        files = request.files.getlist('images')
        for file in files:
            if file and allowed_file(file.filename):
                img = Image.open(io.BytesIO(file.read()))
                images.append(img.convert('RGB'))
    
    # Get additional parameters
    params = {
        'width': int(request.form.get('width', 512)),
        'height': int(request.form.get('height', 512)),
        'steps': int(request.form.get('steps', 30)),
        'guidance_scale': float(request.form.get('guidance_scale', 7.5))
    }
    
    # Generate
    if model_config.get('type') == 'text-to-image':
        result = model_manager.generate_text_to_image(prompt, **params)
    else:
        if not images:
            return jsonify({"error": "At least one image is required for image-to-image"}), 400
        result = model_manager.generate_image_to_image(images, prompt, **params)
    
    # Convert result to base64
    return image_to_response(result)


def handle_json_generation():
    """Handle generation with JSON data (base64 images)."""
    data = request.get_json()
    
    model_id = data.get('model_id')
    prompt = data.get('prompt', '')
    
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    
    if model_id not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model: {model_id}"}), 400
    
    # Load model if not already loaded
    model_config = AVAILABLE_MODELS[model_id]
    if model_manager.current_model_id != model_id:
        model_manager.load_model(model_id, model_config)
    
    # Get images from base64
    images = []
    if 'images' in data:
        for img_data in data['images']:
            if img_data.startswith('data:image'):
                img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img.convert('RGB'))
    
    # Get additional parameters
    params = {
        'width': int(data.get('width', 512)),
        'height': int(data.get('height', 512)),
        'steps': int(data.get('steps', 30)),
        'guidance_scale': float(data.get('guidance_scale', 7.5))
    }
    
    # Generate
    if model_config.get('type') == 'text-to-image':
        result = model_manager.generate_text_to_image(prompt, **params)
    else:
        if not images:
            return jsonify({"error": "At least one image is required for image-to-image"}), 400
        result = model_manager.generate_image_to_image(images, prompt, **params)
    
    # Convert result to base64
    return image_to_response(result)


def image_to_response(image: Image.Image):
    """Convert PIL Image to JSON response with base64 data."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return jsonify({
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload an image and return its path/ID."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Generate unique filename
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_name = f"{timestamp}_{filename}"
    filepath = os.path.join(UPLOAD_DIR, unique_name)
    
    # Save file
    file.save(filepath)
    
    return jsonify({
        "success": True,
        "filename": unique_name,
        "path": f"/uploads/{unique_name}"
    })


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded files."""
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


# Serve frontend static files
@app.route('/')
def index():
    """Serve the main frontend page."""
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
    return send_file(os.path.join(frontend_path, 'index.html'))


@app.route('/assets/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    logger.info(f"=== SERVE_STATIC CALLED ===")
    logger.info(f"Serving static file: {filename} from path: {static_path}")
    logger.info(f"Static path exists: {os.path.exists(static_path)}")
    logger.info(f"Requested file exists: {os.path.exists(os.path.join(static_path, filename))}")
    logger.info(f"Full file path: {os.path.join(static_path, filename)}")
    return send_from_directory(static_path, filename)


if __name__ == '__main__':
    logger.info(f"Starting AI Image Editor server on {HOST}:{PORT}")
    logger.info(f"Models directory: {os.path.join(os.path.dirname(__file__), '..', 'models')}")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
