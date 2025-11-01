import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from diffusers import StableDiffusionXLPipeline, KandinskyV22Pipeline, KandinskyV22Img2ImgPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class SketchToImageGenerator:
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models = {}
        self.current_model = None
        
    def _setup_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self, model_type: str = "Stable Diffusion 2.1"):
        """Load the specified model"""
        if model_type in self.models:
            self.current_model = self.models[model_type]
            return
        
        try:
            if model_type == "Stable Diffusion 2.1":
                model_id = "stabilityai/stable-diffusion-2-1"
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            elif model_type == "Stable Diffusion XL":
                model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16"
                )
            elif model_type == "Kandinsky 2.2":
                from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
                
                prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
                    "kandinsky-community/kandinsky-2-2-prior",
                    torch_dtype=torch.float16
                )
                pipe = KandinskyV22Pipeline.from_pretrained(
                    "kandinsky-community/kandinsky-2-2-decoder",
                    torch_dtype=torch.float16
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            pipe = pipe.to(self.device)
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                pipe.enable_xformers_memory_efficient_attention()
            
            self.models[model_type] = pipe
            self.current_model = pipe
            logger.info(f"Loaded model: {model_type}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {str(e)}")
            raise
    
    def generate(self, 
                 sketch: Image.Image, 
                 prompt: str,
                 model_type: str = "Stable Diffusion 2.1",
                 negative_prompt: str = "",
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 50,
                 num_images: int = 1,
                 strength: float = 0.8) -> List[Image.Image]:
        """Generate images from sketch"""
        
        self.load_model(model_type)
        
        # Prepare the sketch
        sketch = sketch.convert("RGB")
        sketch = sketch.resize((512, 512))
        
        try:
            if model_type == "Kandinsky 2.2":
                return self._generate_kandinsky(sketch, prompt, negative_prompt, 
                                              guidance_scale, num_inference_steps, num_images)
            else:
                return self._generate_stable_diffusion(sketch, prompt, negative_prompt,
                                                     guidance_scale, num_inference_steps, num_images, strength)
        
        except Exception as e:
            logger.error(f"Error generating images: {str(e)}")
            raise
    
    def _generate_stable_diffusion(self, sketch, prompt, negative_prompt, guidance_scale, steps, num_images, strength):
        """Generate using Stable Diffusion img2img"""
        from diffusers import StableDiffusionImg2ImgPipeline
        
        # Convert to img2img pipeline if needed
        if not isinstance(self.current_model, StableDiffusionImg2ImgPipeline):
            self.current_model = StableDiffusionImg2ImgPipeline(**self.current_model.components)
        
        results = self.current_model(
            prompt=[prompt] * num_images,
            image=[sketch] * num_images,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
            num_images_per_prompt=1
        )
        
        return results.images
    
    def _generate_kandinsky(self, sketch, prompt, negative_prompt, guidance_scale, steps, num_images):
        """Generate using Kandinsky 2.2"""
        from diffusers import KandinskyV22Img2ImgPipeline
        
        if not isinstance(self.current_model, KandinskyV22Img2ImgPipeline):
            # For Kandinsky, we need both prior and decoder
            prior_pipe = self.models["Kandinsky 2.2"][0]
            decoder_pipe = self.models["Kandinsky 2.2"][1]
            
            img_emb = prior_pipe(
                prompt=prompt,
                num_inference_steps=steps,
                num_images_per_prompt=num_images
            )
            
            negative_emb = prior_pipe(
                prompt=negative_prompt or "",
                num_inference_steps=steps,
                num_images_per_prompt=num_images
            ) if negative_prompt else None
            
            results = decoder_pipe(
                image=sketch,
                image_embeds=img_emb.image_embeds,
                negative_image_embeds=negative_emb.image_embeds if negative_emb else None,
                height=512,
                width=512,
                num_inference_steps=steps,
                strength=0.8
            )
            
            return results.images
        else:
            return self.current_model(
                prompt=[prompt] * num_images,
                image=[sketch] * num_images,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=[negative_prompt] * num_images if negative_prompt else None
            ).images
    
    def generate_variations(self, image: Image.Image, num_variations: int = 4) -> List[Image.Image]:
        """Generate variations of an existing image"""
        prompt = "variation of the image, same style and composition"
        return self.generate(image, prompt, num_images=num_variations, strength=0.3)