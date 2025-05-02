import torch
import numpy as np
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
from data.transform import get_transforms
import natsort
import glob
from PIL import Image
import os


def normalize(band, lower_percent=2, upper_percent=98):
    """
    Apply percentile stretching to enhance contrast, only considering valid pixels.

    Args:
        band: Input image band as numpy array
        lower_percent: Lower percentile boundary (default 2%)
        upper_percent: Upper percentile boundary (default 98%)

    Returns:
        Normalized band with values in [0, 1]
    """
    # Create mask for valid pixels
    valid_mask = (band > 0)

    # If no valid pixels, return zeros
    if not np.any(valid_mask):
        return np.zeros_like(band, dtype=np.float32)

    # Extract valid pixels for percentile calculation
    valid_pixels = band[valid_mask]
    # Calculate percentiles based only on valid pixels
    lower = np.percentile(valid_pixels, lower_percent)
    upper = np.percentile(valid_pixels, upper_percent)

    # Create a copy to avoid modifying the original
    result = band.copy().astype(np.float32)

    # Apply stretching only to valid pixels
    result[valid_mask] = np.clip((band[valid_mask] - lower) / (upper - lower), 0, 1)

    # Set invalid pixels to 0
    result[~valid_mask] = 0

    return result


def read_images(product_paths):

    images = []

    for path in product_paths:

        data = Image.open(path)
        data = np.array(data)
        data = normalize(data)

        images.append(data)



    images = np.dstack(images)
    return images


class Sentinel2Dataset(Dataset):
    def __init__(self, df_x, df_y,
                 train,
                 augmentation,
                 img_size):

        self.df_x = df_x
        self.df_y = df_y
        self.train = train
        self.augmentation = augmentation
        self.img_size = img_size
        # self.transform = get_transforms(train=self.train,
        #                                 augmentation=self.augmentation)

    def __getitem__(self, index):
        # Load images
        x_paths = natsort.natsorted(glob.glob(os.path.join(self.df_x["path"][index], "*.png"), recursive=False))
        x_data = read_images(x_paths)
        x_data = cv2.resize(x_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x_data = torch.from_numpy(x_data).float()
        x_data = torch.permute(x_data, (2, 0, 1))  # HWC to CHW

        y_paths = natsort.natsorted(glob.glob(os.path.join(self.df_x["path"][index], "*.png"), recursive=False))
        y_data = read_images(y_paths)
        y_data = cv2.resize(y_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        y_data = torch.from_numpy(y_data).float()
        y_data = torch.permute(y_data, (2, 0, 1))  # HWC to CHW

        # transformed = self.transform(image=x_data, mask=y_data)
        # y_data = transformed["mask"]
        # x_data = transformed["image"]

        return x_data, y_data

    def __len__(self):
        return len(self.df_x)


class Sentinel2TCIDataset(Dataset):
    def __init__(self, df_path,
                 train,
                 augmentation,
                 img_size):

        self.df_path = df_path
        self.train = train
        self.augmentation = augmentation
        self.img_size = img_size
        self.transform = get_transforms(train=self.train,
                                        augmentation=self.augmentation)

    def __getitem__(self, index):
        # Load images
        x_path = self.df_path.l1c_path.iloc[index]
        x_data = cv2.imread(x_path)
        x_data = cv2.cvtColor(x_data, cv2.COLOR_BGR2RGB)
        x_data = cv2.resize(x_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x_data = np.array(x_data).astype(np.float32) / 255.0
        x_data = torch.from_numpy(x_data).float()
        x_data = torch.permute(x_data, (2, 0, 1))  # HWC to CHW

        y_path = self.df_path.l2a_path.iloc[index]
        y_data = cv2.imread(y_path)
        y_data = cv2.cvtColor(y_data, cv2.COLOR_BGR2RGB)
        y_data = cv2.resize(y_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        y_data = np.array(y_data).astype(np.float32) / 255.0
        y_data = torch.from_numpy(y_data).float()
        y_data = torch.permute(y_data, (2, 0, 1))  # HWC to CHW

        # transformed = self.transform(image=x_data, mask=y_data)
        # y_data = transformed["mask"]
        # x_data = transformed["image"]


        return x_data, y_data

    def __len__(self):
        return len(self.df_path)

