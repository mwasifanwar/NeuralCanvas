import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Optional
import logging
import os

logger = logging.getLogger(__name__)

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class NeuralStyleTransfer:
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.vgg = VGG19().to(self.device).eval()
        self.style_weights = [1.0, 0.8, 0.5, 0.3, 0.1]
        self.predefined_styles = self._load_predefined_styles()
        
    def _setup_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_predefined_styles(self) -> Dict[str, str]:
        """Load predefined style images"""
        styles = {
            "oil_painting": "styles/oil_painting.jpg",
            "watercolor": "styles/watercolor.jpg", 
            "anime": "styles/anime.jpg",
            "cyberpunk": "styles/cyberpunk.jpg",
            "fantasy": "styles/fantasy.jpg",
            "impressionist": "styles/impressionist.jpg",
            "abstract": "styles/abstract.jpg"
        }
        return styles
    
    def _load_image(self, image_path: str, size: int = 512) -> torch.Tensor:
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        return self._preprocess_image(image, size)
    
    def _preprocess_image(self, image: Image.Image, size: int = 512) -> torch.Tensor:
        """Preprocess image for VGG"""
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def _deprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL Image"""
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.squeeze(0)),
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                               std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Lambda(lambda x: x.clamp(0, 1)),
            transforms.ToPILImage()
        ])
        return transform(tensor.cpu())
    
    def _compute_gram_matrix(self, x):
        """Compute Gram matrix for style representation"""
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(b * c * h * w)
    
    def transfer_style(self, 
                      content_image: Image.Image,
                      style_image: Optional[Image.Image] = None,
                      style_name: Optional[str] = None,
                      strength: float = 0.7,
                      num_steps: int = 300,
                      content_weight: float = 1.0,
                      style_weight: float = 1000.0) -> Image.Image:
        """Perform neural style transfer"""
        
        # Load style image
        if style_name and style_name in self.predefined_styles:
            style_path = self.predefined_styles[style_name]
            if os.path.exists(style_path):
                style_tensor = self._load_image(style_path)
            else:
                raise FileNotFoundError(f"Style image not found: {style_path}")
        elif style_image:
            style_tensor = self._preprocess_image(style_image)
        else:
            raise ValueError("Either style_image or style_name must be provided")
        
        # Preprocess content image
        content_tensor = self._preprocess_image(content_image)
        
        # Initialize generated image
        generated = content_tensor.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.Adam([generated], lr=0.01)
        
        # Get feature representations
        content_features = self.vgg(content_tensor)
        style_features = self.vgg(style_tensor)
        
        # Compute style Gram matrices
        style_grams = [self._compute_gram_matrix(f) for f in style_features]
        
        for step in range(num_steps):
            generated_features = self.vgg(generated)
            
            # Content loss
            content_loss = F.mse_loss(generated_features[3], content_features[3])
            
            # Style loss
            style_loss = 0
            for gen_feat, style_gram, weight in zip(generated_features, style_grams, self.style_weights):
                gen_gram = self._compute_gram_matrix(gen_feat)
                style_loss += weight * F.mse_loss(gen_gram, style_gram)
            
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                logger.info(f"Step {step}, Total loss: {total_loss.item():.4f}")
        
        # Apply strength parameter
        if strength < 1.0:
            generated = strength * generated + (1 - strength) * content_tensor
        
        return self._deprocess_image(generated)
    
    def fast_style_transfer(self,
                           content_image: Image.Image,
                           style_name: str,
                           strength: float = 0.7) -> Image.Image:
        """Fast style transfer using pre-trained models"""
        try:
            # This would use a pre-trained fast style transfer model
            # For now, we'll use the standard method but with fewer steps
            return self.transfer_style(
                content_image=content_image,
                style_name=style_name,
                strength=strength,
                num_steps=100  # Faster but lower quality
            )
        except Exception as e:
            logger.error(f"Fast style transfer failed: {str(e)}")
            # Fall back to original image
            return content_image