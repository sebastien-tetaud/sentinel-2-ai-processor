import sys
import os
import pystac_client
from dotenv import load_dotenv

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from loguru import logger
import time

def create_cdse_query_url(
    collection_name="SENTINEL-2",
    product_type="MSIL2A",
    polygon=None,
    start_interval=None,
    end_interval=None,
    max_cloud_cover=100,
    max_items=1000,
    additional_filters=None,
    orderby="ContentDate/Start"  # Add orderby parameter with default value
):
    """
    Create a query URL for the Copernicus Data Space Ecosystem OData API.

    Parameters:
    -----------
    collection_name : str
        The collection name (e.g., 'SENTINEL-2', 'SENTINEL-1')
    product_type : str
        The product type (e.g., 'MSIL2A', 'MSIL1C', 'GRD')
    polygon : str
        WKT polygon string for spatial filtering
    start_interval : str
        Start time in ISO format with Z for UTC (e.g., '2023-01-01T00:00:00.000Z')
    end_interval : str
        End time in ISO format with Z for UTC (e.g., '2023-01-31T23:59:59.999Z')
    max_cloud_cover : int
        Maximum cloud cover percentage (0-100)
    max_items : int
        Maximum number of items to return
    additional_filters : list
        List of additional filter strings to add to the query
    orderby : str or None
        Field to order results by (e.g., 'ContentDate/Start', 'ContentDate/Start desc')
        Set to None to skip ordering

    Returns:
    --------
    str
        Complete URL for the OData API query
    """

    # Basic filter for collection
    filter_parts = [f"Collection/Name eq '{collection_name}'"]

    # Add spatial filter if provided
    if polygon:
        filter_parts.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')")

    # Add product type filter
    if product_type:
        filter_parts.append(f"contains(Name,'{product_type}')")

    # Add temporal filters if provided
    if start_interval:
        filter_parts.append(f"ContentDate/Start gt {start_interval}")
    if end_interval:
        filter_parts.append(f"ContentDate/Start lt {end_interval}")

    # Add cloud cover filter if applicable
    # Only add for optical sensors (Sentinel-2)
    if collection_name == 'SENTINEL-2' and max_cloud_cover < 100:
        filter_parts.append(
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and "
            f"att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover})"
        )

    # Add any additional filters
    if additional_filters:
        filter_parts.extend(additional_filters)

    # Construct the URL with all filters
    filter_string = " and ".join(filter_parts)
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter={filter_string}"

    # Add orderby parameter if specified
    if orderby:
        url += f"&$orderby={orderby}"

    # Add top parameter for limiting results
    url += f"&$top={max_items}"

    return url

# Set up loguru logger
log_filename = f"sentinel_query_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Remove the default sink and add custom ones
logger.remove()
# Add a sink for the file with the format you want
logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
# Add a sink for stdout with a simpler format
logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")

# Define your bounding box and date range
bbox = [146.5, -22.0, 149.5, -20.0]
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 1, 1)
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
            # Append to the overall results list
            all_l1c_results.extend(l1c_results)
            all_l2a_results.extend(l2a_results)
        else:
            logger.warning(f"Mismatch in counts for {date_interval}: L1C={l1c_count}, L2A={l2a_count}")

        # Print results for this interval
        logger.info(f"L1C Items for {date_interval}: {l1c_count}")
        logger.info(f"L2A Items for {date_interval}: {l2a_count}")
        logger.info("####")

    except Exception as e:
        logger.error(f"Error processing interval {date_interval}: {str(e)}")

    # Move to the next 5-day interval
    current_date = next_date
    time.sleep(1)  # Sleep for 1 second to avoid overwhelming the API

# Create DataFrames from the collected results
df_l1c = pd.DataFrame(all_l1c_results)
df_l2a = pd.DataFrame(all_l2a_results)

# Log final counts
logger.success(f"Query completed. Total L1C items: {len(df_l1c)}, Total L2A items: {len(df_l2a)}")
logger.info(f"Log saved to {log_filename}")

# Save DataFrames to CSV
csv_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
l1c_csv = f"sentinel_l1c_data_{csv_timestamp}.csv"
l2a_csv = f"sentinel_l2a_data_{csv_timestamp}.csv"

df_l1c.to_csv(l1c_csv, index=False)
df_l2a.to_csv(l2a_csv, index=False)
logger.success(f"Data saved to {l1c_csv} and {l2a_csv}")
