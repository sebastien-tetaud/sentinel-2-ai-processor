import torch
import numpy as np
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
from data.transform import get_transforms

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
