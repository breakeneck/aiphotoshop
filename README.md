# AI Image Editor

A web application for AI-powered image generation and editing using quantized models.

## Features

- **Text to Image**: Generate images from text prompts using Stable Diffusion models
- **Image to Image**: Edit existing images using vision-language models (Qwen)
- **Dynamic Model Loading**: Models are loaded on-demand, only one at a time to save memory
- **Multiple Image Support**: Upload multiple images for image-to-image editing
- **Modern Web Interface**: Clean, responsive UI with drag-and-drop support

## Project Structure

```
aiphotoshop/
├── backend/
│   ├── app.py           # Flask API server
│   ├── config.py        # Configuration and model definitions
│   └── model_manager.py # Model loading and inference
├── frontend/
│   └── index.html       # Main HTML page
├── static/
│   ├── style.css        # Styles
│   └── app.js           # Frontend JavaScript
├── models/              # Place your quantized models here
│   ├── qwen-image/      # Qwen vision-language model
│   ├── sdxl/            # Stable Diffusion XL
│   └── sdturbo/         # Stable Diffusion Turbo
├── uploads/             # Uploaded images (auto-created)
├── main.py              # Entry point
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your quantized models in the `models/` directory:
   - `models/qwen-image/` - For Qwen vision-language model
   - `models/sdxl/` - For Stable Diffusion XL
   - `models/sdturbo/` - For Stable Diffusion Turbo

## Running

```bash
python main.py
```

Then open http://localhost:5000 in your browser.

## API Endpoints

### GET /api/models
Returns list of available models.

### GET /api/status
Returns current model status.

### POST /api/load_model
Load a specific model.
```json
{
    "model_id": "qwen-image"
}
```

### POST /api/unload_model
Unload current model.

### POST /api/generate
Generate or edit an image.

**Text to Image:**
```json
{
    "model_id": "sd-xl",
    "prompt": "A beautiful sunset over mountains",
    "width": 512,
    "height": 512,
    "steps": 30
}
```

**Image to Image:**
```json
{
    "model_id": "qwen-image",
    "prompt": "Make this image look like a painting",
    "images": ["data:image/png;base64,..."]
}
```

### POST /api/upload
Upload an image file.

## Adding Custom Models

1. Place your model files in the `models/` directory
2. Edit `backend/config.py` to add your model configuration:

```python
AVAILABLE_MODELS = {
    "my-model": {
        "name": "My Custom Model",
        "type": "text-to-image",  # or "image-to-image"
        "path": os.path.join(MODELS_DIR, "my-model"),
        "description": "Description of your model",
        "supports_multiple_images": False
    }
}
```

## Supported Model Types

### Text to Image
- Stable Diffusion XL
- Stable Diffusion Turbo
- Any diffusers-compatible model

### Image to Image
- Qwen2-VL (supports multiple images)
- Other vision-language models

## Notes

- Only one model is loaded at a time to conserve memory
- Models are loaded on-demand when selected from the dropdown
- GPU acceleration is automatically used if CUDA is available
- For quantized models, ensure you have enough VRAM or use CPU mode
