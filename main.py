import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
from pathlib import Path
import time

from core.sketch_to_image import SketchToImageGenerator
from core.style_transfer import NeuralStyleTransfer
from core.image_enhancer import ImageEnhancer
from core.model_manager import ModelManager
from utils.image_processing import preprocess_sketch, resize_image, convert_to_sketch
from utils.config import load_config, save_config

# Page configuration
st.set_page_config(
    page_title="NeuralCanvas - AI Digital Art Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'sketch_generator' not in st.session_state:
        st.session_state.sketch_generator = None
    if 'style_transfer' not in st.session_state:
        st.session_state.style_transfer = None
    if 'image_enhancer' not in st.session_state:
        st.session_state.image_enhancer = None
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'current_sketch' not in st.session_state:
        st.session_state.current_sketch = None

def load_models():
    """Load AI models on demand"""
    with st.spinner("🔄 Loading AI models... This may take a moment."):
        if st.session_state.sketch_generator is None:
            st.session_state.sketch_generator = SketchToImageGenerator()
        if st.session_state.style_transfer is None:
            st.session_state.style_transfer = NeuralStyleTransfer()
        if st.session_state.image_enhancer is None:
            st.session_state.image_enhancer = ImageEnhancer()

def main():
    st.title("🎨 NeuralCanvas - AI Digital Art Studio - Wasif")
    st.markdown("Transform your sketches into stunning photorealistic artwork with AI magic!")
    
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "AI Model",
            ["Stable Diffusion 2.1", "Stable Diffusion XL", "Kandinsky 2.2"],
            help="Choose the AI model for image generation"
        )
        
        # Style selection
        style_options = {
            "Realistic": "photorealistic",
            "Oil Painting": "oil_painting", 
            "Watercolor": "watercolor",
            "Anime": "anime",
            "Cyberpunk": "cyberpunk",
            "Fantasy": "fantasy"
        }
        selected_style = st.selectbox("Art Style", list(style_options.keys()))
        
        # Enhancement options
        enhance_quality = st.checkbox("Enhance Image Quality", value=True)
        apply_style_transfer = st.checkbox("Apply Style Transfer", value=False)
        
        # Generation parameters
        st.subheader("Generation Parameters")
        guidance_scale = st.slider("Creativity", 1.0, 20.0, 7.5, 0.5)
        num_inference_steps = st.slider("Detail Level", 10, 100, 50, 10)
        num_images = st.slider("Number of Images", 1, 4, 2)
        
        # Style transfer strength
        if apply_style_transfer:
            style_strength = st.slider("Style Strength", 0.1, 1.0, 0.7, 0.1)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Draw Sketch", "📁 Upload Sketch", "🖼️ Style Gallery", "📊 Generated Art"])
    
    with tab1:
        st.header("Draw Your Sketch")
        st.markdown("Use the canvas below to draw your sketch, then generate amazing artwork!")
        
        # Drawing canvas
        from streamlit_drawable_canvas import st_canvas
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=3,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=400,
            width=512,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if canvas_result.image_data is not None:
            # Convert canvas to sketch
            sketch_image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
            st.session_state.current_sketch = sketch_image
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(sketch_image, caption="Your Sketch", use_column_width=True)
            
            with col2:
                if st.button("✨ Generate Art from Sketch", type="primary"):
                    generate_art_from_sketch(
                        sketch_image, 
                        model_choice,
                        style_options[selected_style],
                        guidance_scale,
                        num_inference_steps,
                        num_images,
                        enhance_quality,
                        apply_style_transfer,
                        style_strength if apply_style_transfer else 0.7
                    )
    
    with tab2:
        st.header("Upload Sketch")
        st.markdown("Upload an existing sketch or image to transform")
        
        uploaded_file = st.file_uploader(
            "Choose a sketch/image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload your sketch or image"
        )
        
        if uploaded_file is not None:
            sketch_image = Image.open(uploaded_file).convert('RGB')
            sketch_image = resize_image(sketch_image, 512)
            
            # Option to convert to sketch
            convert_to_sketch_option = st.checkbox("Convert image to sketch", value=True)
            if convert_to_sketch_option:
                sketch_image = convert_to_sketch(sketch_image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(sketch_image, caption="Input Sketch", use_column_width=True)
            
            with col2:
                if st.button("✨ Generate Art from Upload", type="primary"):
                    generate_art_from_sketch(
                        sketch_image,
                        model_choice,
                        style_options[selected_style],
                        guidance_scale,
                        num_inference_steps,
                        num_images,
                        enhance_quality,
                        apply_style_transfer,
                        style_strength if apply_style_transfer else 0.7
                    )
    
    with tab3:
        st.header("Style Gallery")
        st.markdown("Browse and select from pre-defined art styles")
        
        style_cols = st.columns(3)
        style_images = {
            "Realistic": "styles/realistic.jpg",
            "Oil Painting": "styles/oil_painting.jpg", 
            "Watercolor": "styles/watercolor.jpg",
            "Anime": "styles/anime.jpg",
            "Cyberpunk": "styles/cyberpunk.jpg",
            "Fantasy": "styles/fantasy.jpg"
        }
        
        for idx, (style_name, style_path) in enumerate(style_images.items()):
            with style_cols[idx % 3]:
                if Path(style_path).exists():
                    st.image(style_path, caption=style_name, use_column_width=True)
                else:
                    st.info(f"Style: {style_name}")
                
                if st.button(f"Use {style_name}", key=f"style_{idx}"):
                    st.session_state.selected_style = style_name
                    st.success(f"Selected {style_name} style!")
    
    with tab4:
        st.header("Generated Artwork")
        st.markdown("Your recently generated artwork")
        
        if st.session_state.generated_images:
            cols = st.columns(2)
            for idx, (sketch, generated, metadata) in enumerate(st.session_state.generated_images[-4:]):
                with cols[idx % 2]:
                    st.image(generated, caption=f"Artwork {idx+1}", use_column_width=True)
                    with st.expander("Details"):
                        st.write(f"Model: {metadata['model']}")
                        st.write(f"Style: {metadata['style']}")
                        st.write(f"Guidance: {metadata['guidance_scale']}")
                        
                        # Download button
                        img_bytes = generated.tobytes()
                        st.download_button(
                            label="Download Image",
                            data=img_bytes,
                            file_name=f"neuralcanvas_art_{idx+1}.png",
                            mime="image/png",
                            key=f"download_{idx}"
                        )
        else:
            st.info("No artwork generated yet. Create something amazing!")

def generate_art_from_sketch(sketch, model_choice, style, guidance_scale, steps, num_images, enhance, style_transfer, style_strength):
    """Generate artwork from sketch with the given parameters"""
    load_models()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Preparing sketch...")
        processed_sketch = preprocess_sketch(sketch)
        progress_bar.progress(10)
        
        status_text.text("🎨 Generating artwork with AI...")
        
        # Generate base images
        generated_images = st.session_state.sketch_generator.generate(
            sketch=processed_sketch,
            prompt=f"{style} style, high quality, detailed",
            model_type=model_choice,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            num_images=num_images
        )
        progress_bar.progress(60)
        
        # Apply style transfer if requested
        if style_transfer and st.session_state.style_transfer:
            status_text.text("🔄 Applying style transfer...")
            styled_images = []
            for img in generated_images:
                styled_img = st.session_state.style_transfer.transfer_style(
                    content_image=img,
                    style_name=style,
                    strength=style_strength
                )
                styled_images.append(styled_img)
            generated_images = styled_images
            progress_bar.progress(80)
        
        # Enhance image quality
        if enhance and st.session_state.image_enhancer:
            status_text.text("✨ Enhancing image quality...")
            enhanced_images = []
            for img in generated_images:
                enhanced_img = st.session_state.image_enhancer.enhance(img)
                enhanced_images.append(enhanced_img)
            generated_images = enhanced_images
            progress_bar.progress(95)
        
        # Store results
        for gen_img in generated_images:
            metadata = {
                "model": model_choice,
                "style": style,
                "guidance_scale": guidance_scale,
                "steps": steps,
                "timestamp": time.time()
            }
            st.session_state.generated_images.append((sketch, gen_img, metadata))
        
        progress_bar.progress(100)
        status_text.text("✅ Artwork generation complete!")
        
        # Display results
        st.success("🎉 Your artwork has been generated!")
        cols = st.columns(min(2, len(generated_images)))
        for idx, img in enumerate(generated_images):
            with cols[idx % len(cols)]:
                st.image(img, caption=f"Generated Art {idx+1}", use_column_width=True)
    
    except Exception as e:
        st.error(f"❌ Error generating artwork: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")

if __name__ == "__main__":
    main()