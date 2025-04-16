import albumentations as albu
from albumentations.pytorch import ToTensorV2

def get_transforms(train=False, augmentation=False):
    """
    Returns transformation pipeline for Sentinel-2 imagery

    Args:
        train (bool): Whether to use training augmentations

    Returns:
        albumentations.Compose: Transformation pipeline
    """
    if train:
        # Training transforms with augmentations
        return albu.Compose([

            ToTensorV2()
        ])
    else:
        # Validation/Test transforms (minimal)
        return albu.Compose([

            ToTensorV2()
        ])