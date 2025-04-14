import os
import gradio as gr
import numpy as np
import io
import random
from PIL import Image
from dotenv import load_dotenv
import pystac_client
from datetime import datetime
from src.auth.auth import S3Connector
from src.utils.utils import extract_s3_path_from_url
from src.utils.stac_client import get_product_content

# Load environment variables
load_dotenv()

# Get credentials from environment variables
ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")
ENDPOINT_URL = 'https://eodata.dataspace.copernicus.eu'
ENDPOINT_STAC = "https://stac.dataspace.copernicus.eu/v1/"
BUCKET_NAME = "eodata"

# Initialize the connector
connector = S3Connector(
    endpoint_url=ENDPOINT_URL,
    access_key_id=ACCESS_KEY_ID,
    secret_access_key=SECRET_ACCESS_KEY,
    region_name='default'
)

# Connect to S3
s3 = connector.get_s3_resource()
s3_client = connector.get_s3_client()
buckets = connector.list_buckets()
print("Available buckets:", buckets)
catalog = pystac_client.Client.open(ENDPOINT_STAC)


def fetch_sentinel_image(longitude, latitude, date_from, date_to, cloud_cover):
    """Fetch a Sentinel image based on criteria."""
    try:
        # Use the coordinates from inputs
        LON, LAT = float(longitude), float(latitude)

        # Use the date range from inputs
        date_range = f"{date_from}/{date_to}"

        cloud_query = f"eo:cloud_cover<{cloud_cover}"

        items_txt = catalog.search(
            collections=['sentinel-2-l2a'],
            intersects=dict(type="Point", coordinates=[LON, LAT]),
            datetime=date_range,
            query=[cloud_query]
        ).item_collection()

        if len(items_txt) == 0:
            return None, f"No images found for the specified criteria at coordinates ({LON}, {LAT}) with cloud cover < {cloud_cover}%."

        # Randomly select an image from the available items
        selected_item = random.choice(items_txt)

        # Format datetime for readability
        datetime_str = selected_item.properties.get('datetime', 'N/A')
        try:
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            formatted_date = datetime_str

        # Extract metadata for display
        metadata = f"""
        ## Product Information
        - **Location**: {LAT}°N, {LON}°E
        - **Date**: {formatted_date}
        - **Cloud Cover**: {selected_item.properties.get('eo:cloud_cover', 'N/A')}%
        - **Cloud Cover Threshold**: < {cloud_cover}%
        - **Satellite**: {selected_item.properties.get('platform', 'N/A')}
        - **Product ID**: {selected_item.id}
        - **Items Found**: {len(items_txt)} matching products
        """

        # Get the TCI_60m asset from the randomly selected item
        product_url = extract_s3_path_from_url(selected_item.assets['TCI_60m'].href)

        product_content = get_product_content(s3_client=s3_client, bucket_name=BUCKET_NAME,
                                     object_url=product_url)
        print(f"Selected product URL: {product_url}")


        # Convert to PIL Image
        img = Image.open(io.BytesIO(product_content))

        return img, metadata

    except ValueError as ve:
        error_message = f"Invalid input: {str(ve)}. Please ensure longitude and latitude are valid numbers."
        print(error_message)
        return None, error_message
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)
        return None, error_message


# Create Gradio interface
with gr.Blocks(title="Sentinel Product Viewer") as demo:
    gr.Markdown("# Sentinel-2 Product Viewer")
    gr.Markdown("Browse and view Sentinel-2 satellite product")

    with gr.Row():
        with gr.Column(scale=1):
            # Location inputs
            with gr.Row():
                longitude = gr.Number(label="Longitude", value=15.0, minimum=-180, maximum=180)
                latitude = gr.Number(label="Latitude", value=50.0, minimum=-90, maximum=90)

            # Date range inputs
            with gr.Row():
                date_from = gr.Textbox(label="Date From (YYYY-MM-DD)", value="2024-05-01")
                date_to = gr.Textbox(label="Date To (YYYY-MM-DD)", value="2025-02-01")

            # Cloud cover slider
            cloud_cover = gr.Slider(
                label="Max Cloud Cover (%)",
                minimum=0,
                maximum=100,
                value=50,
                step=5
            )

            # Diverse landscape location buttons
            gr.Markdown("### Diverse Locations")
            with gr.Row():
                italy_btn = gr.Button("Italy")
                amazon_btn = gr.Button("Amazon Rainforest")
            with gr.Row():
                tokyo_btn = gr.Button("Tokyo")
                great_barrier_btn = gr.Button("Great Barrier Reef")

            with gr.Row():
                iceland_btn = gr.Button("Iceland Glacier")
                canada_btn = gr.Button("Baffin Island")




            fetch_btn = gr.Button("Fetch Random Image", variant="primary")

        with gr.Column(scale=2):
            image_output = gr.Image(type="pil", label="Sentinel-2 Image")
            metadata_output = gr.Markdown(label="Image Metadata")

    # Button click handlers for diverse landscapes
    italy_btn.click(lambda: (12.39, 42.05), outputs=[longitude, latitude])
    amazon_btn.click(lambda: (-64.7, -3.42), outputs=[longitude, latitude])
    tokyo_btn.click(lambda: (139.70, 35.65), outputs=[longitude, latitude])
    great_barrier_btn.click(lambda: (150.97, -20.92), outputs=[longitude, latitude])

    iceland_btn.click(lambda: (-18.17, 64.61), outputs=[longitude, latitude])
    # rice_terraces_btn.click(lambda: (121.1, 16.9), outputs=[longitude, latitude])
    canada_btn.click(lambda: (-71.56, 67.03), outputs=[longitude, latitude])

    # Main search button
    fetch_btn.click(
        fn=fetch_sentinel_image,
        inputs=[longitude, latitude, date_from, date_to, cloud_cover],
        outputs=[image_output, metadata_output]
    )

    gr.Markdown("## About")
    gr.Markdown("""
    This application allows you to browse and view Sentinel-2 satellite imagery using the Copernicus Data Space Ecosystem.

    - **Location**: Enter longitude and latitude coordinates or select distinctive landscapes
    - **TCI Images**: The images shown are true color (RGB) composites at 60m resolution
    - **Date Range**: Specify the date range to search for images
    - **Cloud Cover**: Adjust the maximum acceptable cloud cover percentage
    - **Random Selection**: A random image that matches the criteria will be selected for display
    """)

if __name__ == "__main__":
    demo.launch(share=True)
