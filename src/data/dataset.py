import torch
import numpy as np
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset


class Sentinel2Dataset(Dataset):

    def __init__(self, df_path, train, augmentation):

        self.df_path = df_path
        self.train = train
        self.augmentation = augmentation

    def __getitem__(self, index):

        x_path = self.df_path.l1c_path.iloc[index]
        x_data = cv2.imread(x_path)
        x_data = np.array(x_data)

        y_path = self.df_path.l2a_path.iloc[index]
        y_data = cv2.imread(y_path)
        # y_data = cv2.cvtColor(y_data, cv2.COLOR_BGR2RGB)
        y_data = np.array(y_data)
        # y_data = np.transpose(y_data,(2,1,0))

        return x_data, y_data

    def __len__(self):
        return len(self.df_path)