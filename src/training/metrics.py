# Save this to /home/ubuntu/project/sentinel-2-ai-processor/src/training/metrics.py

import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from typing import Dict, List, Optional, Tuple

class SpectralAngularMapper(nn.Module):
    """
    Spectral Angular Mapper (SAM) metric for measuring spectral similarity.
    SAM = arccos(dot(a,b) / (||a|| * ||b||))
    """

    def __init__(self):
        super(SpectralAngularMapper, self).__init__()
        self.total_sam = 0.0
        self.count = 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SAM between prediction and target.

        Args:
            pred: Predictions tensor
            target: Ground truth tensor

        Returns:
            SAM value (lower is better)
        """
        # Flatten spatial dimensions
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)

        # Compute dot product
        dot_product = torch.sum(pred_flat * target_flat, dim=1)

        # Compute norms
        pred_norm = torch.norm(pred_flat, dim=1)
        target_norm = torch.norm(target_flat, dim=1)

        # Avoid division by zero
        eps = 1e-8
        cos_sim = dot_product / (pred_norm * target_norm + eps)

        # Clip cosine similarity to [-1, 1] to avoid numerical issues
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # Compute angle in radians and convert to degrees
        angle = torch.acos(cos_sim) * (180.0 / torch.pi)

        # Return mean angle across batch
        return angle.mean()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state with new predictions and targets."""
        sam_value = self.forward(pred, target)
        self.total_sam += sam_value.item() * pred.size(0)
        self.count += pred.size(0)

    def reset(self) -> None:
        """Reset metric state."""
        self.total_sam = 0.0
        self.count = 0

    def compute(self) -> torch.Tensor:
        """Compute final result."""
        return torch.tensor(self.total_sam / max(self.count, 1))

    def result(self) -> torch.Tensor:
        """Alias for compute() for backward compatibility."""
        return self.compute()

class MultiSpectralMetrics:
    """
    Class for computing and tracking multiple metrics across spectral bands.
    Handles valid pixel masking for satellite imagery.
    """

    def __init__(self, bands: List[str], device: str = 'cuda'):
        """
        Initialize metrics for each spectral band.

        Args:
            bands: List of band names
            device: Device to compute metrics on
        """
        self.bands = bands
        self.device = device
        self.metrics = {}

        # Initialize all metrics for all bands
        for band in bands:
            self.metrics[band] = {
                'psnr': PeakSignalNoiseRatio(data_range=1.0).to(device),
                'rmse': MeanSquaredError(squared=False).to(device),
                'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
                'sam': SpectralAngularMapper().to(device)
            }

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics for all bands.

        Args:
            outputs: Model predictions [B, C, H, W]
            targets: Ground truth [B, C, H, W]
        """
        for c, band in enumerate(self.bands):
            # Extract channel data
            outputs_c = outputs[:, c:c+1, :, :]  # Keep dim for SSIM
            targets_c = targets[:, c:c+1, :, :]

            # Create channel-wise valid mask (True for valid pixels)
            valid_mask_c = (targets_c >= 0)

            # Update metrics that work with tensors directly
            if torch.any(valid_mask_c):
                # For metrics that need 2D (dropping channel dim)
                outputs_valid_c = outputs_c[valid_mask_c]
                targets_valid_c = targets_c[valid_mask_c]

                self.metrics[band]['psnr'].update(outputs_valid_c, targets_valid_c)
                self.metrics[band]['rmse'].update(outputs_valid_c, targets_valid_c)

                # For metrics that need 4D (keeping batch and channel dims)
                if valid_mask_c.all():
                    # If all pixels are valid, use tensors as is for SSIM
                    self.metrics[band]['ssim'].update(outputs_c, targets_c)
                    self.metrics[band]['sam'].update(outputs_c, targets_c)
                else:
                    # Handle partial valid pixels for metrics requiring full images
                    # This requires creating masked versions that replace invalid pixels
                    masked_outputs = outputs_c.clone()
                    masked_targets = targets_c.clone()
                    # Replace invalid pixels with zeros (or another approach)
                    masked_outputs[~valid_mask_c] = 0
                    masked_targets[~valid_mask_c] = 0

                    self.metrics[band]['ssim'].update(masked_outputs, masked_targets)
                    self.metrics[band]['sam'].update(masked_outputs, masked_targets)

    def compute(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics for all bands.

        Returns:
            Dictionary with metrics for each band
        """
        results = {}
        for band in self.bands:
            results[band] = {}
            for metric_name, metric in self.metrics[band].items():
                results[band][metric_name] = metric.compute().item()
        return results

    def reset(self) -> None:
        """Reset all metrics."""
        for band in self.bands:
            for metric in self.metrics[band].values():
                metric.reset()
