import torch


def masked_mse_loss(predictions, targets, valid_mask):
    """
    Compute MSE loss only on valid pixels.

    Args:
        predictions: Model predictions (B, C, H, W)
        targets: Ground truth images (B, C, H, W)
        valid_mask: Binary mask indicating valid pixels (B, 1, H, W) or (B, C, H, W)

    Returns:
        Masked MSE loss computed only on valid pixels
    """
    # Apply mask
    masked_pred = predictions * valid_mask
    masked_target = targets * valid_mask

    # Count valid pixels for normalization
    num_valid_pixels = torch.sum(valid_mask) + 1e-8

    # Compute MSE loss on valid pixels only
    loss = torch.sum((masked_pred - masked_target) ** 2) / num_valid_pixels

    return loss