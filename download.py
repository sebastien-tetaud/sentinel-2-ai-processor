
import os
import pystac_client
from dotenv import load_dotenv

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from loguru import logger
import time
from dotenv import load_dotenv

from src.auth.auth import S3Connector
from src.utils.utils import extract_s3_path_from_url
from loguru import logger


import xml.etree.ElementTree as ET
import pandas as pd
import os
from typing import Optional


def parse_safe_manifest(content: str) -> Optional[pd.DataFrame]:
    """
    Parse a Sentinel SAFE manifest file and extract href attributes.

    Args:
        manifest_path (str): Path to the manifest.safe file

    Returns:
        pd.DataFrame: DataFrame containing href values and file information,
                     or None if an error occurred
    """
    try:

        # Parse the content
        root = ET.fromstring(content)

        # Extract all elements with an href attribute using a generic approach
        hrefs = []
        for elem in root.findall(".//*[@href]"):
            href = elem.get('href')
            if href:
                hrefs.append(href)

        # Create DataFrame with href values and file information
        df_files = pd.DataFrame({
            'href': hrefs,
            'file_type': [href.split('.')[-1] if '.' in href else 'unknown' for href in hrefs],
            'file_name': [os.path.basename(href) for href in hrefs]
        })


        return df_files

    except ET.ParseError as e:
        logger.error(f"XML parsing error : {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing manifest: {str(e)}")
        return None


def filter_band_files(df_files, bands=None, product_type=None, resolution=None):
    """
    Filter a dataframe for Sentinel-2 band files supporting both L1C and L2A formats.

    Args:
        df_files (pd.DataFrame): DataFrame with 'href' column containing file paths
        bands (list, optional): List of band names to filter for (e.g., ['B02', 'B03', 'B04']).
                               If None, defaults to RGB bands.
        product_type (str, optional): Product type ('L1C' or 'L2A'). If None, both types are included.
        resolution (str or int, optional): Specific resolution to filter for L2A products ('10m', '20m', '60m' or 10, 20, 60).
                                         If None, includes all resolutions.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only requested band files
    """
    # Define default bands to filter if not specified
    if bands is None:
        bands = ['B02', 'B03', 'B04']  # RGB bands by default

    # Convert resolution to string if it's an integer
    if resolution is not None:
        resolution = str(resolution)

    # Build regex patterns to match both L1C and L2A formats
    band_patterns = []

    for band in bands:
        # L1C format: IMG_DATA/*_B02.jp2
        if product_type is None or product_type.upper() == 'L1C':
            band_patterns.append(r'IMG_DATA/.*_' + band + r'\.jp2')

        # L2A formats with correct pattern: IMG_DATA/R20m/T55KGR_20200103T001101_B02_20m.jp2
        if product_type is None or product_type.upper() == 'L2A':
            if resolution:
                # If specific resolution is provided, filter for that resolution
                band_patterns.append(r'IMG_DATA/R' + resolution + r'm/.*_' + band + r'_' + resolution + r'm\.jp2')
            else:
                # If no resolution is specified, include all resolutions
                band_patterns.extend([
                    r'IMG_DATA/R10m/.*_' + band + r'_10m\.jp2',
                    r'IMG_DATA/R20m/.*_' + band + r'_20m\.jp2',
                    r'IMG_DATA/R60m/.*_' + band + r'_60m\.jp2'
                ])

    filter_condition = False
    for pattern in band_patterns:
        filter_condition = filter_condition | df_files['href'].str.contains(pattern, regex=True)

    df_gr = df_files[filter_condition].copy()  # Create a copy to avoid the warning


    # Remove leading ./ from href paths
    df_gr['href'] = df_gr['href'].str.replace(r'^\./', '', regex=True)

    return df_gr


def download_bands(s3_client, bucket, bucket_name, df, bands, product_type, resolution, output_dir, max_attempts=10, retry_delay=10):
    """
    Download Sentinel-2 band files from S3 based on dataframe information.

    Args:
        s3_client: S3 client object
        bucket: S3 bucket object
        bucket_name: Name of the S3 bucket
        df (pd.DataFrame): DataFrame with 'S3Path' column containing S3 paths
        bands (list): List of bands to download
        product_type (str): Product type ('L1C' or 'L2A')
        resolution (int, optional): Resolution in meters. Required for L2A products.
        output_dir (str): Base directory to save files
        max_attempts (int): Maximum number of download attempts
        retry_delay (int): Delay between retry attempts in seconds
    """

    for index, row in df.iterrows():
        # Extract base S3 URL
        s3_base_url = extract_s3_path_from_url(row['S3Path']).replace("/eodata","")
        s3_manifest_url = f"{s3_base_url}/manifest.safe"
        _, filename = os.path.split(s3_manifest_url)

        # Try to download manifest file with retry logic
        attempt = 0
        content = None

        while attempt < max_attempts:
            try:
                # Get the manifest file
                response = s3_client.get_object(Bucket=bucket_name, Key=s3_manifest_url)
                # Check if successful
                if response["ResponseMetadata"]['HTTPStatusCode'] == 200:
                    content = response['Body'].read()
                    logger.info(f"Downloaded manifest from {s3_manifest_url}")
                    break
                else:
                    logger.warning(f"Unexpected status: {response['ResponseMetadata']['HTTPStatusCode']}")
                    attempt += 1
                    time.sleep(retry_delay)

            except Exception as e:
                logger.warning(f"Error downloading manifest: {str(e)}")
                attempt += 1
                time.sleep(retry_delay)

        if content is None:
            logger.error(f"Failed to download manifest after {max_attempts} attempts, skipping this product")
            continue

        df_tmp = parse_safe_manifest(content=content)
        df_bands = filter_band_files(df_tmp, bands=bands, product_type=product_type, resolution=resolution)

        for gr in df_bands['href']:
            # Create full S3 URL for the band file
            band_s3_url = f"{s3_base_url}/{gr}"

            # Extract just the filename from the path
            filename = os.path.basename(gr)

            # Extract product ID for folder structure
            path_safe = s3_base_url.split(os.sep)[7].replace(".SAFE","")
            path_save = os.path.join(output_dir, path_safe)
            os.makedirs(path_save, exist_ok=True)

            # Download the file with retry logic
            attempt = 0

            while attempt < max_attempts:
                try:
                    # Download the band file
                    bucket.download_file(band_s3_url, f"{path_save}/{filename}")
                    logger.info(f"Downloaded {filename} to {path_save}")
                    break
                except Exception as e:
                    logger.warning(f"Error downloading band file {filename}: {str(e)}")
                    attempt += 1
                    if attempt < max_attempts:
                        logger.info(f"Retrying download of {filename}, attempt {attempt+1} of {max_attempts}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to download {filename} after {max_attempts} attempts")


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")
    ENDPOINT_URL = 'https://eodata.dataspace.copernicus.eu'
    ENDPOINT_STAC = "https://stac.dataspace.copernicus.eu/v1/"
    DATASET_VERSION = "V1"
    BUKETNAME = "eodata"
    BASE_DIR = f"/mnt/disk/dataset/sentinel-ai-processor"
    DATASET_DIR = f"{BASE_DIR}/{DATASET_VERSION}"

    connector = S3Connector(
        endpoint_url=ENDPOINT_URL,
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        region_name='default'
    )
    # Get S3 client and resource from the connector instance
    s3 = connector.get_s3_resource()
    s3_client = connector.get_s3_client()
    buckets = connector.list_buckets()
    bucket = s3.Bucket(BUKETNAME)


    input_dir = os.path.join(DATASET_DIR, "input")
    output_dir = os.path.join(DATASET_DIR, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Set up loguru logger
    log_filename = f"{DATASET_DIR}/sentinel_download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # Remove the default sink and add custom ones
    logger.remove()
    # Add a sink for the file with the format you want
    logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    # Add a sink for stdout with a simpler format
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")

    df_l1c = pd.read_csv(f"{DATASET_DIR}/input_l1c.csv")
    df_l2a = pd.read_csv(f"{DATASET_DIR}/output_l2a.csv")


    bands=['TCI']
    # download_bands(s3_client=s3_client, bucket=bucket, bucket_name=BUKETNAME, df=df_l2a,
    #                product_type="L2A", bands=bands, resolution=60, output_dir=output_dir,
    #                max_attempts=10, retry_delay=10)

    download_bands(s3_client=s3_client, bucket=bucket, bucket_name=BUKETNAME, df=df_l1c,
                   product_type="L1C", bands=bands, resolution=None, output_dir=input_dir,
                   max_attempts=10, retry_delay=10)