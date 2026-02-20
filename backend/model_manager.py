"""
Model Manager - handles loading, unloading and inference for AI models.
Only one model is loaded at a time to save memory.
Supports GGUF models via llama-cpp-python.
"""

import os
import gc
import torch
from typing import Optional, Dict, Any, List
from PIL import Image
import io
import base64
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages AI models - loads/unloads models on demand, only one at a time."""
    
    def __init__(self):
        self.current_model = None
        self.current_model_id = None
        self.current_model_type = None
        self.current_model_format = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ModelManager initialized. Device: {self.device}")
    
    def unload_current_model(self):
        """Unload the current model to free memory."""
        if self.current_model is not None:
            logger.info(f"Unloading model: {self.current_model_id}")
            del self.current_model
            self.current_model = None
            self.current_model_id = None
            self.current_model_type = None
            self.current_model_format = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded and memory cleared")
    
    def load_model(self, model_id: str, model_config: Dict[str, Any]):
        """Load a specific model, unloading any existing model first."""
        if self.current_model_id == model_id:
            logger.info(f"Model {model_id} already loaded")
            return True
        
        # Unload current model
        self.unload_current_model()
        
        try:
            model_type = model_config.get("type")
            model_path = model_config.get("path")
            model_format = model_config.get("format", "transformers")
            
            logger.info(f"Loading model: {model_id} (type: {model_type}, format: {model_format})")
            
            if model_format == "gguf":
                self._load_gguf_model(model_id, model_path, model_config)
            elif model_type == "image-to-image":
                self._load_image_to_image_model(model_id, model_path)
            elif model_type == "text-to-image":
                self._load_text_to_image_model(model_id, model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.current_model_id = model_id
            self.current_model_type = model_type
            self.current_model_format = model_format
            logger.info(f"Model {model_id} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            self.current_model = None
            self.current_model_id = None
            self.current_model_type = None
            self.current_model_format = None
            raise
    
    def _load_gguf_model(self, model_id: str, model_path: str, model_config: Dict[str, Any]):
        """Load a GGUF model using llama-cpp-python."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            from llama_cpp import Llama
            
            # Determine if we have GPU support
            n_gpu_layers = -1 if self.device == "cuda" else 0
            
            # Load the model
            logger.info(f"Loading GGUF model from {model_path}")
            self.current_model = {
                "llm": Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=4096,  # Context window
                    verbose=True
                ),
                "config": model_config
            }
            logger.info("GGUF model loaded successfully")
            
        except ImportError as e:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install llama-cpp-python\n"
                "For GPU support: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )
    
    def _load_image_to_image_model(self, model_id: str, model_path: str):
        """Load an image-to-image model (e.g., Qwen Image)."""
        # Check if model path exists
        if not os.path.exists(model_path):
            # Try to load from HuggingFace or use a placeholder
            logger.warning(f"Model path {model_path} not found. Attempting to load from HuggingFace...")
        
        try:
            # Try to load Qwen2-VL or similar vision-language model
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            self.current_model = {
                "model": Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path if os.path.exists(model_path) else "Qwen/Qwen2-VL-7B-Instruct",
                    torch_dtype="auto",
                    device_map="auto"
                ),
                "processor": AutoProcessor.from_pretrained(
                    model_path if os.path.exists(model_path) else "Qwen/Qwen2-VL-7B-Instruct"
                ),
                "process_vision_info": process_vision_info
            }
        except ImportError:
            logger.warning("Qwen VL not available, trying alternative...")
            # Fallback to a simpler approach or placeholder
            self.current_model = {"type": "placeholder", "model_id": model_id}
    
    def _load_text_to_image_model(self, model_id: str, model_path: str):
        """Load a text-to-image model (e.g., Stable Diffusion)."""
        try:
            from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
            
            if "sdxl" in model_id.lower():
                self.current_model = StableDiffusionXLPipeline.from_pretrained(
                    model_path if os.path.exists(model_path) else "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            else:
                self.current_model = StableDiffusionPipeline.from_pretrained(
                    model_path if os.path.exists(model_path) else "stabilityai/stable-diffusion-2-1",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            if self.device == "cuda":
                self.current_model = self.current_model.to("cuda")
                
        except ImportError:
            logger.warning("Diffusers not available, using placeholder...")
            self.current_model = {"type": "placeholder", "model_id": model_id}
    
    def generate_image_to_image(
        self, 
        images: List[Image.Image], 
        prompt: str,
        **kwargs
    ) -> Image.Image:
        """Generate image from input images and prompt."""
        if self.current_model_type != "image-to-image":
            raise ValueError("Current model is not an image-to-image model")
        
        if self.current_model is None:
            raise ValueError("No model loaded")
        
        logger.info(f"Generating image-to-image with prompt: {prompt[:50]}...")
        
        # Handle GGUF model
        if self.current_model_format == "gguf":
            return self._generate_with_gguf(images, prompt, **kwargs)
        
        # Handle placeholder model
        if isinstance(self.current_model, dict) and self.current_model.get("type") == "placeholder":
            return self._placeholder_generation(images[0] if images else None, prompt)
        
        try:
            model = self.current_model["model"]
            processor = self.current_model["processor"]
            process_vision_info = self.current_model["process_vision_info"]
            
            # Prepare messages for Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img} for img in images
                    ] + [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            
            # Generate
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # For now, return a placeholder image with the response text
            # In a real implementation, you might use a separate image generation model
            return self._create_text_image(output_text[0] if output_text else "No output")
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def _generate_with_gguf(
        self, 
        images: List[Image.Image], 
        prompt: str,
        **kwargs
    ) -> Image.Image:
        """Generate using GGUF model."""
        llm = self.current_model["llm"]
        
        # Convert images to base64 for the prompt
        image_descriptions = []
        for i, img in enumerate(images):
            # Resize if too large
            if img.width > 512 or img.height > 512:
                img = img.copy()
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            # Create a description placeholder
            image_descriptions.append(f"[Image {i+1}: {img.width}x{img.height}]")
        
        # Build the prompt
        full_prompt = f"""You are an AI image editing assistant. The user has provided {len(images)} image(s) and wants you to help with image editing.

Images provided:
{chr(10).join(image_descriptions)}

User request: {prompt}

Please provide a detailed description of how you would edit the image based on the user's request. Be creative and specific.

Response:"""
        
        logger.info(f"Sending prompt to GGUF model...")
        
        # Generate response
        response = llm(
            full_prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        
        output_text = response.get("choices", [{}])[0].get("text", "No response generated")
        logger.info(f"Model response: {output_text[:200]}...")
        
        # Return an image with the response text
        return self._create_text_image(output_text, images[0] if images else None)
    
    def generate_text_to_image(self, prompt: str, **kwargs) -> Image.Image:
        """Generate image from text prompt."""
        if self.current_model_type != "text-to-image":
            raise ValueError("Current model is not a text-to-image model")
        
        if self.current_model is None:
            raise ValueError("No model loaded")
        
        logger.info(f"Generating text-to-image with prompt: {prompt[:50]}...")
        
        # Handle placeholder model
        if isinstance(self.current_model, dict) and self.current_model.get("type") == "placeholder":
            return self._placeholder_generation(None, prompt)
        
        try:
            # Stable Diffusion generation
            image = self.current_model(
                prompt=prompt,
                num_inference_steps=kwargs.get("steps", 30),
                guidance_scale=kwargs.get("guidance_scale", 7.5),
                width=kwargs.get("width", 512),
                height=kwargs.get("height", 512)
            ).images[0]
            
            return image
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def _placeholder_generation(self, base_image: Optional[Image.Image], prompt: str) -> Image.Image:
        """Create a placeholder result when models aren't available."""
        # Create a simple placeholder image
        if base_image:
            img = base_image.copy()
        else:
            img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        
        return img
    
    def _create_text_image(self, text: str, base_image: Optional[Image.Image] = None) -> Image.Image:
        """Create an image with text (for text responses from VLM)."""
        from PIL import ImageDraw, ImageFont
        
        # Use base image or create new one
        if base_image:
            img = base_image.copy().convert('RGB')
            # Resize if needed
            if img.width < 512:
                img = img.resize((512, int(512 * img.height / img.width)), Image.Resampling.LANCZOS)
        else:
            img = Image.new('RGB', (512, 512), color=(30, 30, 50))
        
        # Create semi-transparent overlay for text
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 180))
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        
        draw = ImageDraw.Draw(img)
        
        # Use default font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
        
        # Wrap text
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) > 60:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        
        # Draw text with background
        y = 20
        for line in lines[:30]:  # Limit lines
            # Draw text background
            bbox = draw.textbbox((15, y), line, font=font)
            draw.rectangle([bbox[0]-5, bbox[1]-2, bbox[2]+5, bbox[3]+2], fill=(0, 0, 0, 200))
            # Draw text
            draw.text((15, y), line, fill=(255, 255, 255), font=font)
            y += 20
        
        return img.convert('RGB')
    
    def get_status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            "current_model": self.current_model_id,
            "model_type": self.current_model_type,
            "model_format": self.current_model_format,
            "device": self.device,
            "loaded": self.current_model is not None
        }


# Global model manager instance
model_manager = ModelManager()
