import os
from datetime import datetime, timedelta
import pystac_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

# Define constants
ENDPOINT_URL = 'https://eodata.dataspace.copernicus.eu'
ENDPOINT_STAC = "https://stac.dataspace.copernicus.eu/v1/"
# bbox = [146.5, -22.0, 149.5, -20.0]
# bbox = [-5.2, 41.3, 9.8, 51.2]
bbox = [3.2833, 45.3833, 11.2, 50.1833]


# Open the STAC catalog
catalog = pystac_client.Client.open(ENDPOINT_STAC)

# Define the start and end dates
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 1, 1)

# Loop through the date range with a step of 5 days
current_date = start_date
while current_date < end_date:
    # Calculate the end of the current 5-day interval
    next_date = min(current_date + timedelta(days=5), end_date)

    # Format the dates as required for the query
    date_interval = f"{current_date.strftime('%Y-%m-%d')}/{next_date.strftime('%Y-%m-%d')}"

    # Search for L2A and L1C items in the current interval
    items_l2a = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_interval,
        query=["eo:cloud_cover<=100"]
    ).item_collection()

    items_l1c = catalog.search(
        collections=["sentinel-2-l1c"],
        bbox=bbox,
        datetime=date_interval,
        query=["eo:cloud_cover<=100"]
    ).item_collection()
    # if len(items_l1c)!=len(items_l2a):
    # Print or process items_l1c if needed
    print(f"L1C Items for {date_interval}: {len(items_l1c)}")
    print(f"L2A Items for {date_interval}: {len(items_l2a)}")
    print("####")
    # # Save items_l2a to a GeoJSON file for the current interval
    # if len(items_l2a) > 0:
    #     output_file = f"items_l2a_{date_interval}.geojson"
    #     items_l2a.save_object(include_self_link=False, dest_href=output_file)
    #     print(f"Saved L2A items to {output_file}")
    # else:
    #     print(f"No L2A items found for {date_interval}")

    # Move to the next interval
    current_date = next_date
