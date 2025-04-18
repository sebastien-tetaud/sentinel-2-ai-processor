#TODO

# PSNR
# SSIM for each band
# Spectral Angle Mapper  https://lightning.ai/docs/torchmetrics/stable/image/spectral_angle_mapper.html
# RMSE

import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

class BandMetrics:
    def __init__(self, num_bands, data_range=1.0, device):
        self.psnr_channels = [PeakSignalNoiseRatio(data_range=data_range).to(device) for _ in range(num_bands)]
        self.ssim_channels = [StructuralSimilarityIndexMeasure(data_range=data_range).to(device) for _ in range(num_bands)]

    def reset(self):
        for psnr, ssim in zip(self.psnr_channels, self.ssim_channels):
            psnr.reset()
            ssim.reset()

    def update(self, outputs, y_data, valid_mask):
        for c, (psnr, ssim) in enumerate(zip(self.psnr_channels, self.ssim_channels)):
            outputs_c = outputs[:, c, :, :]
            y_c = y_data[:, c, :, :]
            valid_mask_c = (y_c >= 0)
            outputs_valid_c = outputs_c[valid_mask_c]
            y_valid_c = y_c[valid_mask_c]
            psnr.update(outputs_valid_c, y_valid_c)
            ssim.update(outputs_c.unsqueeze(1), y_c.unsqueeze(1))  # SSIM expects input with shape (N, C, H, W)

    def compute(self):
        psnr_values = [psnr.compute().item() for psnr in self.psnr_channels]
        ssim_values = [ssim.compute().item() for ssim in self.ssim_channels]
        return psnr_values, ssim_values
