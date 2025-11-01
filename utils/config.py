import yaml
from pathlib import Path
from typing import Dict, Any
import os

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Create default config
        default_config = {
            "models": {
                "stable_diffusion_2_1": {
                    "enabled": True,
                    "default": True
                },
                "stable_diffusion_xl": {
                    "enabled": True,
                    "default": False
                },
                "kandinsky_2_2": {
                    "enabled": False,
                    "default": False
                }
            },
            "generation": {
                "default_steps": 50,
                "default_guidance": 7.5,
                "max_images": 4,
                "image_size": 512
            },
            "style_transfer": {
                "enabled": True,
                "default_strength": 0.7
            },
            "enhancement": {
                "super_resolution": True,
                "color_correction": True,
                "denoising": True
            },
            "ui": {
                "theme": "light",
                "canvas_size": 512,
                "show_tutorial": True
            }
        }
        save_config(default_config, config_path)
        return default_config
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    """Save configuration to YAML file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for specific model"""
    config = load_config()
    return config.get("models", {}).get(model_name, {})

def update_config(section: str, key: str, value: Any):
    """Update configuration value"""
    config = load_config()
    
    if section not in config:
        config[section] = {}
    
    config[section][key] = value
    save_config(config)

def get_default_generation_params() -> Dict[str, Any]:
    """Get default generation parameters"""
    config = load_config()
    generation = config.get("generation", {})
    
    return {
        "steps": generation.get("default_steps", 50),
        "guidance_scale": generation.get("default_guidance", 7.5),
        "num_images": generation.get("max_images", 4),
        "image_size": generation.get("image_size", 512)
    }