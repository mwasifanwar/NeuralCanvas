import torch
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        
        # Model configurations
        self.model_configs = {
            "stable_diffusion_2_1": {
                "repo_id": "stabilityai/stable-diffusion-2-1",
                "files": ["model_index.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"],
                "type": "diffusion"
            },
            "stable_diffusion_xl": {
                "repo_id": "stabilityai/stable-diffusion-xl-base-1.0", 
                "files": ["model_index.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"],
                "type": "diffusion"
            },
            "real_esrgan": {
                "repo_id": "xinntao/Real-ESRGAN",
                "files": ["RealESRGAN_x4plus.pth"],
                "type": "enhancement"
            }
        }
    
    def download_model(self, model_name: str) -> str:
        """Download model if not already present"""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_dir / model_name
        config = self.model_configs[model_name]
        
        if model_path.exists():
            logger.info(f"Model {model_name} already exists at {model_path}")
            return str(model_path)
        
        logger.info(f"Downloading model {model_name}...")
        try:
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            logger.info(f"Model {model_name} downloaded successfully")
            return str(model_path)
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {str(e)}")
            raise
    
    def load_model(self, model_name: str, force_reload: bool = False):
        """Load model into memory"""
        if model_name in self.loaded_models and not force_reload:
            return self.loaded_models[model_name]
        
        model_path = self.download_model(model_name)
        config = self.model_configs[model_name]
        
        try:
            if config["type"] == "diffusion":
                from diffusers import StableDiffusionPipeline
                model = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            elif config["type"] == "enhancement":
                # Real-ESRGAN loading handled separately
                model = None
            else:
                raise ValueError(f"Unsupported model type: {config['type']}")
            
            self.loaded_models[model_name] = model
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def unload_model(self, model_name: str):
        """Unload model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Model {model_name} unloaded")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_configs.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model"""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        model_path = self.model_dir / model_name
        
        return {
            "name": model_name,
            "repo_id": config["repo_id"],
            "type": config["type"],
            "downloaded": model_path.exists(),
            "loaded": model_name in self.loaded_models,
            "path": str(model_path)
        }