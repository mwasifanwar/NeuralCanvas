import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from typing import Tuple, Optional

def preprocess_sketch(sketch: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """Preprocess sketch for AI generation"""
    # Convert to RGB if necessary
    if sketch.mode != 'RGB':
        sketch = sketch.convert('RGB')
    
    # Resize to target size
    sketch = sketch.resize(target_size, Image.LANCZOS)
    
    # Enhance contrast for better results
    sketch = ImageOps.autocontrast(sketch, cutoff=2)
    
    return sketch

def convert_to_sketch(image: Image.Image, 
                     blur_radius: int = 2, 
                     threshold: int = 150) -> Image.Image:
    """Convert image to sketch-like appearance"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Invert image
    image = ImageOps.invert(image)
    
    # Apply Gaussian blur
    image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Blend with original
    image = Image.blend(image, ImageOps.invert(image), 0.5)
    
    # Apply threshold
    image = image.point(lambda x: 0 if x < threshold else 255, 'L')
    
    # Convert back to RGB
    return image.convert('RGB')

def resize_image(image: Image.Image, 
                size: int, 
                maintain_aspect: bool = True) -> Image.Image:
    """Resize image while optionally maintaining aspect ratio"""
    if maintain_aspect:
        # Calculate new dimensions maintaining aspect ratio
        ratio = size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, Image.LANCZOS)
    else:
        return image.resize((size, size), Image.LANCZOS)

def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image (basic implementation)"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to create mask
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Invert mask
    mask = 255 - mask
    
    # Apply mask to image
    result = cv2.bitwise_and(img_array, img_array, mask=mask)
    
    return Image.fromarray(result)

def blend_images(foreground: Image.Image, 
                background: Image.Image, 
                alpha: float = 0.7) -> Image.Image:
    """Blend two images together"""
    foreground = foreground.resize(background.size, Image.LANCZOS)
    return Image.blend(background, foreground, alpha)

def add_watermark(image: Image.Image, 
                 text: str = "NeuralCanvas",
                 opacity: int = 128) -> Image.Image:
    """Add watermark to image"""
    from PIL import ImageDraw, ImageFont
    
    # Create a copy to avoid modifying original
    watermarked = image.copy()
    
    # Create drawing context
    draw = ImageDraw.Draw(watermarked)
    
    # Try to use a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position (bottom right)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = (image.width - text_width - 10, image.height - text_height - 10)
    
    # Draw semi-transparent text
    draw.text(position, text, fill=(255, 255, 255, opacity), font=font)
    
    return watermarked

def create_collage(images: list, 
                  cols: int = 2, 
                  spacing: int = 10) -> Image.Image:
    """Create a collage from multiple images"""
    if not images:
        raise ValueError("No images provided for collage")
    
    # Calculate dimensions
    num_images = len(images)
    rows = (num_images + cols - 1) // cols
    
    # Get dimensions of first image
    img_width, img_height = images[0].size
    
    # Calculate collage size
    collage_width = cols * img_width + (cols + 1) * spacing
    collage_height = rows * img_height + (rows + 1) * spacing
    
    # Create blank canvas
    collage = Image.new('RGB', (collage_width, collage_height), 'white')
    
    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        x = col * (img_width + spacing) + spacing
        y = row * (img_height + spacing) + spacing
        
        collage.paste(img, (x, y))
    
    return collage