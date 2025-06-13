import os
import io
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
import pystac_client
from loguru import logger
import warnings

from auth.auth import S3Connector
from utils.utils import extract_s3_path_from_url, load_config
from utils.stac_client import get_product_content
from model_zoo.models import define_model
from utils.torch import load_model_weights
warnings.filterwarnings('ignore')


def initialize_env() -> dict:
    """Load environment variables."""
    try:
        load_dotenv()
        logger.success("Loaded environment variables")
        return {
            "access_key_id": os.environ.get("ACCESS_KEY_ID"),
            "secret_access_key": os.environ.get("SECRET_ACCESS_KEY")
        }
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return {}


def connect_to_s3(endpoint_url: str, access_key_id: str, secret_access_key: str) -> tuple:
    """Connect to S3 storage."""
    try:
        connector = S3Connector(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region_name='default'
        )
        logger.success(f"Successfully connected to {endpoint_url} ")
        return connector.get_s3_resource(), connector.get_s3_client()
    except Exception as e:
        logger.error(f"Failed to connect to S3 storage: {e}")
        return None, None


def fetch_random_item(catalog, bbox: list, start_date: str, end_date: str, max_cloud_cover: int):
    """Fetch a random item from the STAC catalog."""
    try:
        items = catalog.search(
            collections=['sentinel-2-l1c'],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=100
        ).item_collection()
        item = random.choice(items)
        logger.success(f"Random L1C fetched from CDSE: {item.id}")
        return item
    except Exception as e:
        logger.error(f"Failed to fetch a random L1C product: {e}")
        return None


def load_bands_from_s3(s3_client, bucket_name: str, item, bands: list, resize_shape: tuple = (1830, 1830)) -> np.ndarray:
    """Load bands from S3 storage."""
    try:
        band_data = []
        for band_name in bands:
            product_url = extract_s3_path_from_url(item.assets[band_name].href)
            content = get_product_content(s3_client, bucket_name, product_url)
            image = Image.open(io.BytesIO(content)).resize(resize_shape)
            band_data.append(np.array(image))
        logger.success("Loaded bands from S3 storage")
        return np.dstack(band_data)
    except Exception as e:
        logger.error(f"Failed to load bands from S3 storage: {e}")
        return None


def normalize(data_array: np.ndarray) -> tuple:
    """Normalize the data array."""
    try:
        normalized_data, valid_masks = [], []
        for i in range(data_array.shape[2]):
            band = data_array[:, :, i]
            valid_mask = band > 0
            norm_band = band.astype(np.float32)
            norm_band[valid_mask] /= 10000
            norm_band = np.clip(norm_band, 0, 1)
            norm_band[~valid_mask] = 0
            normalized_data.append(norm_band)
            valid_masks.append(valid_mask)
        logger.success("Normalized data array")
        return np.dstack(normalized_data), np.dstack(valid_masks)
    except Exception as e:
        logger.error(f"Failed to normalize data array: {e}")
        return None, None


def preprocess(raw_data: np.ndarray, resize: int, device: torch.device) -> tuple:
    """Preprocess the raw data."""
    try:
        x_data, valid_mask = normalize(raw_data)
        x_data = cv2.resize(x_data, (resize, resize), interpolation=cv2.INTER_AREA)
        valid_mask = cv2.resize(valid_mask.astype(np.uint8), (resize, resize), interpolation=cv2.INTER_NEAREST).astype(bool)
        x_tensor = torch.from_numpy(x_data).float().permute(2, 0, 1).unsqueeze(0).to(device) # [B , C , W, H]
        logger.success("Preprocess raw data successull")
        return x_tensor, valid_mask
    except Exception as e:
        logger.error(f"Failed to preprocess raw data: {e}")
        return None, None


def postprocess(x_tensor: torch.Tensor, pred_tensor: np.ndarray, valid_mask: np.ndarray) -> tuple:
    """Postprocess the prediction."""
    try:
        x_np = x_tensor.cpu().numpy()[0].transpose(1, 2, 0)
        pred_np = pred_tensor
        x_np[~valid_mask] = 0.0
        pred_np[~valid_mask] = 0.0
        logger.success("Postprocess model output successull")
        return x_np, pred_np
    except Exception as e:
        logger.error(f"Failed to postprocess model output: {e}")
        return None, None


def load_model(model_cfg: dict, weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load the model."""
    try:
        model = define_model(
            name=model_cfg["model_name"],
            encoder_name=model_cfg["encoder_name"],
            in_channel=model_cfg["in_channel"],
            out_channels=model_cfg["out_channels"],
            activation=model_cfg["activation"]
        )
        model = load_model_weights(model, filename=weights_path)
        logger.success("Model Loaded")
        return model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def predict(model: torch.nn.Module, x_tensor: torch.Tensor) -> np.ndarray:
    """Make a prediction."""
    try:
        model.eval()
        with torch.no_grad():
            pred = model(x_tensor)
        logger.success("L2A generation successfull")
        return pred.cpu().numpy()[0].transpose(1, 2, 0)
    except Exception as e:
        logger.error(f"Failed to generate L2A: {e}")
        return None


def visualize_results(x_np: np.ndarray, pred_np: np.ndarray, bands: list, cmap: str, output_dir) -> None:
    """Visualize the results."""
    try:
        for idx, band in enumerate(bands):
            fig, axs = plt.subplots(1, 2, figsize=(20, 6))
            axs[0].imshow(x_np[:, :, idx], cmap=cmap)
            axs[0].set_title(f"Input L1C - Band: {band}", fontsize=14)
            axs[0].axis('off')

            axs[1].imshow(pred_np[:, :, idx], cmap=cmap)
            axs[1].set_title(f"Prediction L2A - Band: {band}", fontsize=14)
            axs[1].axis('off')

            fig.savefig(f"{output_dir}/{band}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        logger.success("Visualization generated")
    except Exception as e:
        logger.error(f"Failed to generated visualization: {e}")


def main() -> None:
    # Set up logging
    logger.add("log_AiSen2Core.log", rotation="10 MB")
    logger.info("Start workflow ...")
    # Load environment and configs
    env = initialize_env()
    dir_path = os.getcwd()

    # model_cfg = load_config(f"{dir_path}/cfg/config.yaml")
    # query_cfg = load_config(f"{dir_path}/cfg/query_config.yaml")
    query_cfg = load_config("/home/ubuntu/project/sentinel-2-ai-processor/src/results/2025-05-09_19-12-13/query_config.yaml")
    model_cfg = load_config("/home/ubuntu/project/sentinel-2-ai-processor/src/results/2025-05-09_19-12-13/config.yaml")
    model_path = "/home/ubuntu/project/sentinel-2-ai-processor/src/results/2025-05-09_19-12-13/checkpoints/best_model.pth"
    # model_path = f"{dir_path}/weight/AiSen2Core_EfficientNet_b2.pth"

    # Setup
    endpoint_url = query_cfg["endpoint_url"]
    bucket_name = query_cfg["bucket_name"]
    stac_url = query_cfg["endpoint_stac"]
    s3, s3_client = connect_to_s3(endpoint_url, env["access_key_id"], env["secret_access_key"])
    catalog = pystac_client.Client.open(stac_url)

    # Fetch data
    bands = model_cfg["DATASET"]["bands"]
    bbox = query_cfg["query"]["bbox"]
    start_date = query_cfg["query"]["start_date"]
    end_date = query_cfg["query"]["end_date"]
    max_cloud_cover = query_cfg["query"]["max_cloud_cover"]

    item = fetch_random_item(catalog, bbox, start_date, end_date, max_cloud_cover)

    raw_data = load_bands_from_s3(s3_client, bucket_name, item, bands)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_cfg["MODEL"], model_path, device)

    # Inference
    resize = model_cfg["TRAINING"]["resize"]
    x_tensor, valid_mask = preprocess(raw_data=raw_data, resize=resize, device=device)

    pred_np = predict(model=model, x_tensor=x_tensor)

    x_np, pred_np = postprocess(x_tensor=x_tensor, pred_tensor=pred_np, valid_mask=valid_mask)

    # Visualization
    visualize_results(x_np=x_np, pred_np=pred_np, bands=bands, cmap="Grays_r", output_dir=dir_path)
    logger.success("Workflow completed")


if __name__ == "__main__":
    main()
