import os
import gradio as gr
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

from src.utils.stac_client import download_sentinel_image

# Load environment variables
load_dotenv()
USERNAME = os.environ.get("DESTINE_USERNAME")
PASSWORD = os.environ.get("DESTINE_PASSWORD")

def fetch_sentinel_image(date_from, date_to):
    """Fetch a Sentinel image based on criteria."""
    # Validate date format    
    # Download the image
    content, metadata = download_sentinel_image(
        username=USERNAME,
        password=PASSWORD,
        start_date=date_from,
        end_date= date_to
    )
    
    # Handle error case
    if isinstance(content, str):
        return None, content
    
    # Convert to PIL Image
    try:
        img = Image.open(BytesIO(content))
        
        # Create metadata string
        metadata_str = "\n".join([
            f"**Date:** {metadata.get('datetime', 'Unknown')}",
            # f"**Cloud Cover:** {metadata.get('cloud_cover', 'Unknown')}%",
            f"**Filename:** {metadata.get('filename', 'Unknown')}"
        ])
        
        return img, metadata_str
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Sentinel Image Viewer") as demo:
    gr.Markdown("# Sentinel-2 Image Viewer")
    gr.Markdown("Browse and view Sentinel-2 satellite imagery")
    
    with gr.Row():
        with gr.Column(scale=1):
            date_from = gr.Textbox(label="Date From (YYYY-MM-DD)", value="2024-12-15")
            date_to = gr.Textbox(label="Date To (YYYY-MM-DD)", value="2024-12-30")
            fetch_btn = gr.Button("Fetch Image")
        
        with gr.Column(scale=2):
            image_output = gr.Image(type="pil", label="Sentinel-2 Image")
            metadata_output = gr.Markdown(label="Image Metadata")
    
    fetch_btn.click(
        fn=fetch_sentinel_image, 
        inputs=[date_from, date_to], 
        outputs=[image_output, metadata_output]
    )

    gr.Markdown("## About")
    gr.Markdown("""
    This application allows you to browse and view Sentinel-2 satellite imagery using the DESTINE API.
    
    - **TCI Images**: The images shown are true color (RGB) composites at 60m resolution.
    - **Date Range**: Specify the date range to search for images.
    """)

if __name__ == "__main__":
    demo.launch(share=True)
