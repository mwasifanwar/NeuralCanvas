import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class ImageEnhancer:
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.upscaler = None
        self.face_enhancer = None
        
    def _setup_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_upscaler(self):
        """Load Real-ESRGAN upscaler"""
        if self.upscaler is None:
            try:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                self.upscaler = RealESRGANer(
                    scale=4,
                    model_path='weights/RealESRGAN_x4plus.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=self.device == 'cuda'
                )
            except Exception as e:
                logger.warning(f"Could not load Real-ESRGAN: {str(e)}")
                self.upscaler = None
    
    def enhance(self, 
                image: Image.Image,
                enhance_type: str = "super_resolution",
                scale_factor: int = 2) -> Image.Image:
        """Enhance image quality"""
        
        if enhance_type == "super_resolution":
            return self.super_resolution(image, scale_factor)
        elif enhance_type == "color_correction":
            return self.color_correction(image)
        elif enhance_type == "sharpness":
            return self.sharpness_enhancement(image)
        elif enhance_type == "denoise":
            return self.denoise(image)
        else:
            return image
    
    def super_resolution(self, image: Image.Image, scale_factor: int = 2) -> Image.Image:
        """Apply super-resolution to image"""
        self._load_upscaler()
        
        if self.upscaler is None:
            # Fallback: Use PIL for basic upscaling
            new_size = (image.width * scale_factor, image.height * scale_factor)
            return image.resize(new_size, Image.LANCZOS)
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Enhance with Real-ESRGAN
            enhanced, _ = self.upscaler.enhance(img_array, outscale=scale_factor)
            
            # Convert back to PIL Image
            return Image.fromarray(enhanced)
        
        except Exception as e:
            logger.error(f"Super resolution failed: {str(e)}")
            # Fallback to basic upscaling
            new_size = (image.width * scale_factor, image.height * scale_factor)
            return image.resize(new_size, Image.LANCZOS)
    
    def color_correction(self, image: Image.Image) -> Image.Image:
        """Apply color correction and enhancement"""
        # Convert to HSV for better color manipulation
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.2)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    def sharpness_enhancement(self, image: Image.Image) -> Image.Image:
        """Enhance image sharpness"""
        # Apply unsharp mask filter
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        return image
    
    def denoise(self, image: Image.Image) -> Image.Image:
        """Remove noise from image"""
        # Convert to OpenCV format
        img_array = np.array(image)
        
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        
        # Convert back to PIL Image
        return Image.fromarray(denoised)
    
    def batch_enhance(self, images: List[Image.Image], enhance_type: str = "super_resolution") -> List[Image.Image]:
        """Enhance multiple images"""
        return [self.enhance(img, enhance_type) for img in images]
    
    def auto_enhance(self, image: Image.Image) -> Image.Image:
        """Apply all enhancement techniques automatically"""
        image = self.color_correction(image)
        image = self.sharpness_enhancement(image)
        image = self.denoise(image)
        image = self.super_resolution(image, scale_factor=2)
        return image