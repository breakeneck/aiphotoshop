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
        logger.info(f"=== LOAD MODEL START ===")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Model config: {model_config}")
        
        if self.current_model_id == model_id:
            logger.info(f"Model {model_id} already loaded")
            return True
        
        # Unload current model
        self.unload_current_model()
        
        try:
            model_type = model_config.get("type")
            model_path = model_config.get("path")
            model_format = model_config.get("format", "transformers")
            
            logger.info(f"Loading model: {model_id}")
            logger.info(f"  Type: {model_type}")
            logger.info(f"  Format: {model_format}")
            logger.info(f"  Path: {model_path}")
            logger.info(f"  Device: {self.device}")
            
            if model_format == "gguf":
                logger.info("Loading GGUF model...")
                self._load_gguf_model(model_id, model_path, model_config)
            elif model_format == "transformers" or model_type == "image-to-image":
                logger.info("Loading image-to-image model...")
                self._load_image_to_image_model(model_id, model_path)
            elif model_type == "text-to-image":
                logger.info("Loading text-to-image model...")
                self._load_text_to_image_model(model_id, model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.current_model_id = model_id
            self.current_model_type = model_type
            self.current_model_format = model_format
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
            
            logger.info(f"Model {model_id} loaded successfully")
            logger.info(f"=== LOAD MODEL END ===")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.current_model = None
            self.current_model_id = None
            self.current_model_type = None
            self.current_model_format = None
            raise
    
    def _load_gguf_model(self, model_id: str, model_path: str, model_config: Dict[str, Any]):
        """Load a GGUF model using llama-cpp-python."""
        logger.info(f"=== LOAD GGUF MODEL START ===")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            from llama_cpp import Llama
            
            # Determine if we have GPU support
            n_gpu_layers = -1 if self.device == "cuda" else 0
            logger.info(f"n_gpu_layers: {n_gpu_layers} (device: {self.device})")
            
            # Load the model
            logger.info(f"Loading GGUF model from {model_path}...")
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
            logger.info(f"=== LOAD GGUF MODEL END ===")
            
        except ImportError as e:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install llama-cpp-python\n"
                "For GPU support: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )
    
    def _load_image_to_image_model(self, model_id: str, model_path: str):
        """Load an image-to-image model (e.g., Stable Diffusion img2img)."""
        logger.info(f"=== LOAD IMAGE-TO-IMAGE MODEL START ===")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Device: {self.device}")
        
        # Try to load Stable Diffusion img2img pipeline
        try:
            from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
            
            logger.info("Diffusers library available")
            
            # Check if it's SDXL
            if "sdxl" in model_id.lower() or "xl" in model_id.lower():
                logger.info("Loading SDXL img2img pipeline")
                self.current_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            else:
                logger.info("Loading Stable Diffusion img2img pipeline")
                self.current_model = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
            
            if self.device == "cuda":
                self.current_model = self.current_model.to("cuda")
                logger.info("Model moved to CUDA")
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
            
            logger.info(f"Image-to-image model loaded successfully on {self.device}")
            logger.info(f"=== LOAD IMAGE-TO-IMAGE MODEL END ===")
            
        except ImportError:
            logger.warning("Diffusers not available, trying alternative...")
            # Try to load Qwen2-VL for understanding (but won't edit images)
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                from qwen_vl_utils import process_vision_info
                
                logger.info("Loading Qwen2-VL for image understanding (editing not supported)")
                self.current_model = {
                    "model": Qwen2VLForConditionalGeneration.from_pretrained(
                        "Qwen/Qwen2-VL-7B-Instruct",
                        torch_dtype="auto",
                        device_map="auto"
                    ),
                    "processor": AutoProcessor.from_pretrained(
                        "Qwen/Qwen2-VL-7B-Instruct"
                    ),
                    "process_vision_info": process_vision_info,
                    "type": "vision_language"
                }
                logger.info("Qwen2-VL loaded for image understanding only")
                logger.info(f"=== LOAD IMAGE-TO-IMAGE MODEL END ===")
            except ImportError:
                logger.warning("Qwen VL not available, using placeholder...")
                self.current_model = {"type": "placeholder", "model_id": model_id}
                logger.info(f"=== LOAD IMAGE-TO-IMAGE MODEL END ===")
    
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
        
        # Handle GGUF model - use Stable Diffusion img2img instead
        if self.current_model_format == "gguf":
            logger.warning("GGUF format detected, but using Stable Diffusion img2img for actual image editing")
            # Load Stable Diffusion img2img on the fly
            return self._generate_with_stable_diffusion_img2img(images, prompt, **kwargs)
        
        # Handle placeholder model
        if isinstance(self.current_model, dict) and self.current_model.get("type") == "placeholder":
            logger.warning("Using placeholder model - no actual image editing will occur")
            return self._placeholder_generation(images[0] if images else None, prompt)
        
        # Handle vision-language model (Qwen2-VL) - can't edit images, only understand them
        if isinstance(self.current_model, dict) and self.current_model.get("type") == "vision_language":
            logger.warning("Vision-language model loaded - using Stable Diffusion img2img for actual editing")
            return self._generate_with_stable_diffusion_img2img(images, prompt, **kwargs)
        
        # Handle Stable Diffusion img2img pipeline
        try:
            base_image = images[0] if images else None
            if base_image is None:
                raise ValueError("At least one image is required for image-to-image")
            
            # Resize image if needed
            if base_image.width > 1024 or base_image.height > 1024:
                base_image = base_image.copy()
                base_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Generate with img2img
            strength = kwargs.get("strength", 0.75)  # How much to change the image (0-1)
            num_inference_steps = kwargs.get("steps", 30)
            guidance_scale = kwargs.get("guidance_scale", 7.5)
            
            logger.info(f"Running img2img with strength={strength}, steps={num_inference_steps}")
            
            result = self.current_model(
                prompt=prompt,
                image=base_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            logger.info("Image-to-image generation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def _generate_with_stable_diffusion_img2img(
        self, 
        images: List[Image.Image], 
        prompt: str,
        **kwargs
    ) -> Image.Image:
        """Generate using Stable Diffusion img2img pipeline."""
        logger.info("Loading Stable Diffusion img2img pipeline...")
        
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            
            # Load the pipeline
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            
            if self.device == "cuda":
                pipe = pipe.to("cuda")
            
            logger.info(f"Stable Diffusion img2img loaded on {self.device}")
            
            # Get base image
            base_image = images[0] if images else None
            if base_image is None:
                raise ValueError("At least one image is required for image-to-image")
            
            # Resize image if needed
            if base_image.width > 1024 or base_image.height > 1024:
                base_image = base_image.copy()
                base_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Generate with img2img
            strength = kwargs.get("strength", 0.75)
            num_inference_steps = kwargs.get("steps", 30)
            guidance_scale = kwargs.get("guidance_scale", 7.5)
            
            logger.info(f"Running img2img with strength={strength}, steps={num_inference_steps}")
            
            result = pipe(
                prompt=prompt,
                image=base_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            logger.info("Image-to-image generation completed successfully")
            return result
            
        except ImportError:
            logger.error("Diffusers not available. Install with: pip install diffusers")
            raise ImportError("diffusers library is required for image-to-image generation")
        except Exception as e:
            logger.error(f"Stable Diffusion img2img failed: {str(e)}")
            raise
    
    def _generate_with_gguf(
        self, 
        images: List[Image.Image], 
        prompt: str,
        **kwargs
    ) -> Image.Image:
        """Generate using GGUF model - redirects to Stable Diffusion img2img."""
        logger.warning("GGUF model detected, redirecting to Stable Diffusion img2img")
        return self._generate_with_stable_diffusion_img2img(images, prompt, **kwargs)
    
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
