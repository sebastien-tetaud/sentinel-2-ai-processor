import io
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

# Load environment variables
load_dotenv()

# Now import the module
from src.auth.auth import S3Connector
from src.utils.utils import extract_s3_path_from_url
from src.utils.utils import remove_last_segment_rsplit
from src.utils.cdse_utils import (create_cdse_query_url, get_product,
                                    parse_safe_manifest, download_manifest,
                                    filter_band_files,  download_bands)


ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")
ENDPOINT_URL = 'https://eodata.dataspace.copernicus.eu'
ENDPOINT_STAC = "https://stac.dataspace.copernicus.eu/v1/"
DATASET_VERSION = "V2"
BUCKET_NAME = "eodata"
BASE_DIR = f"/mnt/disk/dataset/sentinel-ai-processor"
DATASET_DIR = f"{BASE_DIR}/{DATASET_VERSION}"
BANDS = ['B02','B03','B04']

connector = S3Connector(
    endpoint_url=ENDPOINT_URL,
    access_key_id=ACCESS_KEY_ID,
    secret_access_key=SECRET_ACCESS_KEY,
    region_name='default')
# Get S3 client and resource from the connector instance
s3 = connector.get_s3_resource()
s3_client = connector.get_s3_client()
buckets = connector.list_buckets()
bucket = s3.Bucket(BUCKET_NAME)

input_dir = os.path.join(DATASET_DIR, "input")
output_dir = os.path.join(DATASET_DIR, "output")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

bbox = [3.2833, 45.3833, 11.2, 50.1833]

log_filename = f"{DATASET_DIR}/sentinel_query_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
# Remove the default sink and add custom ones
logger.remove()
# Add a sink for the file with the format you want
logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
# Add a sink for stdout with a simpler format
logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 15)
max_items = 1000
max_cloud_cover = 100

# Log query parameters
logger.info(f"Query parameters:")
logger.info(f"Bounding box: {bbox}")
logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
logger.info(f"Max items per request: {max_items}")
logger.info(f"Max cloud cover: {max_cloud_cover}%")
# Generate the polygon string from bbox [minx, miny, maxx, maxy]
polygon = f"POLYGON (({bbox[0]} {bbox[1]}, {bbox[0]} {bbox[3]}, {bbox[2]} {bbox[3]}, {bbox[2]} {bbox[1]}, {bbox[0]} {bbox[1]}))"

# Initialize empty lists to store all results
all_l1c_results = []
all_l2a_results = []

# Loop through the date range with a step of 5 days
current_date = start_date
while current_date < end_date:
    # Calculate the end of the current 5-day interval
    next_date = min(current_date + timedelta(days=10), end_date)

    # Format the dates as required for the OData query (ISO format with Z for UTC)
    start_interval = f"{current_date.strftime('%Y-%m-%dT00:00:00.000Z')}"
    end_interval = f"{next_date.strftime('%Y-%m-%dT23:59:59.999Z')}"

    date_interval = f"{current_date.strftime('%Y-%m-%d')}/{next_date.strftime('%Y-%m-%d')}"

    try:

        l2a_query_url = create_cdse_query_url(
            product_type="MSIL2A",
            polygon=polygon,
            start_interval=start_interval,
            end_interval=end_interval,
            max_cloud_cover=max_cloud_cover,
            max_items=max_items,
            orderby="ContentDate/Start"
        )
        # Search for Sentinel-2 L2A products for this interval
        l2a_json = requests.get(l2a_query_url).json()

        # Add interval as metadata to each item
        l2a_results = l2a_json.get('value', [])
        for item in l2a_results:
            item['query_interval'] = date_interval


        l1c_query_url = create_cdse_query_url(
            product_type="MSIL1C",
            polygon=polygon,
            start_interval=start_interval,
            end_interval=end_interval,
            max_cloud_cover=max_cloud_cover,
            max_items=max_items,
            orderby="ContentDate/Start"
        )
        # Search for Sentinel-2 L1C products for this interval
        l1c_json = requests.get(l1c_query_url).json()

        # Add interval as metadata to each item
        l1c_results = l1c_json.get('value', [])
        for item in l1c_results:
            item['query_interval'] = date_interval

        # Count L1C products
        l1c_count = len(l1c_results)
        l2a_count = len(l2a_results)

        if l1c_count == l2a_count:
            # Append to the overall results list?
            all_l1c_results.extend(l1c_results)
            all_l2a_results.extend(l2a_results)
        else:
            logger.warning(f"Mismatch in counts for {date_interval}: L1C={l1c_count}, L2A={l2a_count}")
            all_l1c_results.extend(l1c_results)
            all_l2a_results.extend(l2a_results)

        # Print results for this interval
        logger.info(f"L1C Items for {date_interval}: {l1c_count}")
        logger.info(f"L2A Items for {date_interval}: {l2a_count}")
        logger.info("####")

    except Exception as e:
        logger.error(f"Error processing interval {date_interval}: {str(e)}")
    # Move to the next n-day interval
    current_date = next_date

# Create DataFrames from the collected results
df_l1c = pd.DataFrame(all_l1c_results)
df_l2a = pd.DataFrame(all_l2a_results)
# Select only the required columns
# Select only the required columns
df_l2a = df_l2a[["Name", "S3Path", "Footprint", "GeoFootprint","Attributes"]]
df_l1c = df_l1c[["Name", "S3Path", "Footprint", "GeoFootprint","Attributes"]]

df_l1c['cloud_cover'] = df_l1c['Attributes'].apply(lambda x: x[2]["Value"])
df_l2a['cloud_cover'] = df_l2a['Attributes'].apply(lambda x: x[2]["Value"])

# Create the id_key column based on the function
df_l2a['id_key'] = df_l2a['Name'].apply(remove_last_segment_rsplit)
df_l2a['id_key'] = df_l2a['id_key'].str.replace('MSIL2A_', 'MSIL1C_')  # Replace prefix for matching
df_l1c['id_key'] = df_l1c['Name'].apply(remove_last_segment_rsplit)

# Step 1: Drop duplicates in each DataFrame and keep the first occurrence
df_l2a = df_l2a.drop_duplicates(subset='id_key', keep='first')
df_l1c = df_l1c.drop_duplicates(subset='id_key', keep='first')

# Step 2: Find the common id_keys to ensure both DataFrames have the same order
df_l2a = df_l2a[df_l2a['id_key'].isin(df_l1c['id_key'])]
df_l1c = df_l1c[df_l1c['id_key'].isin(df_l2a['id_key'])]

# Step 3: Align the DataFrames by the order of id_key
df_l2a = df_l2a.set_index('id_key')
df_l1c = df_l1c.set_index('id_key')

df_l2a = df_l2a.loc[df_l1c.index].reset_index()
df_l1c = df_l1c.reset_index()

df_l1c.to_csv(f"{DATASET_DIR}/input_l1c.csv")
df_l2a.to_csv(f"{DATASET_DIR}/output_l2a.csv")

# df_l1c = df_l1c.sample(n=15000, random_state=42)
# df_l2a = df_l2a.sample(n=15000, random_state=42)
df_l1c = df_l1c.reset_index(drop=True)
df_l2a = df_l2a.reset_index(drop=True)

for i in range(min(len(df_l1c), len(df_l2a))):
    if df_l1c['id_key'][i] == df_l2a['id_key'][i]:
        print(f"Match: {df_l1c['id_key'][i]} == {df_l2a['id_key'][i]}")
    else:
        print(f"Mismatch: {df_l1c['id_key'][i]} != {df_l2a['id_key'][i]}")

df_l1c.to_csv(f"{DATASET_DIR}/sample_input_l1c.csv")
df_l2a.to_csv(f"{DATASET_DIR}/sample_output_l2a.csv")

log_filename = f"{DATASET_DIR}/sentinel_download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
# Remove the default sink and add custom ones
logger.remove()
# Add a sink for the file with the format you want
logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
# Add a sink for stdout with a simpler format
logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")


# download_bands(s3_client=s3_client, bucket_name=BUCKET_NAME, df=df_l1c[:1],
#                 product_type="L1C", bands=BANDS, resize=True, resolution=None, output_dir=input_dir,
#                 max_attempts=10, retry_delay=10)

download_bands(s3_client=s3_client, bucket_name=BUCKET_NAME, df=df_l2a[:2],
                product_type="L2A", bands=BANDS, resize=True, resolution=60, output_dir=output_dir,
                max_attempts=10, retry_delay=20)


