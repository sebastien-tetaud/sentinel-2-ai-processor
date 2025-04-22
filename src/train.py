import pandas as pd
import loguru
from loguru import logger
from data.dataset import Sentinel2Dataset
from data.loader import define_loaders
from utils.utils import load_config
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from model_zoo.models import define_model
import torch.optim as optim
from utils.torch import count_parameters, seed_everything, load_model_weights
from training.losses import masked_mse_loss

# Configure loguru logger
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger.add(f"{log_dir}/training.log", rotation="10 MB")

config = load_config(config_path="cfg/config.yaml")
BASE_DIR = config["DATASET"]["base_dir"]
VERSION = config['DATASET']['version']
BANDS = config['DATASET']['bands']
BATCH_SIZE = config['TRAINING']['batch_size']
NUM_WORKERS = config['TRAINING']['num_workers']
RESIZE = config['TRAINING']['resize']
LEARNING_RATE = config['TRAINING']['learning_rate']
NUM_EPOCHS = config['TRAINING']['n_epoch']
SEED = config['TRAINING']['seed']

###
train_path = f"{BASE_DIR}/{VERSION}/train_path.csv"
val_path = f"{BASE_DIR}/{VERSION}/val_path.csv"
test_path = f"{BASE_DIR}/{VERSION}/test_path.csv"
seed_everything(seed=SEED)
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

logger.info(f"Number of training data: {len(df_train)}")
logger.info(f"Number of val data: {len(df_val)}")
logger.info(f"Number of test data: {len(df_test)}")


train_dataset = Sentinel2Dataset(df_path=df_train,
                                 train=True, augmentation=False,
                                 img_size=RESIZE)

val_dataset = Sentinel2Dataset(df_path=df_val,
                               train=False, augmentation=False,
                               img_size=RESIZE)

test_dataset = Sentinel2Dataset(df_path=df_test,
                                 train=True, augmentation=False,
                                 img_size=RESIZE)

train_loader, val_loader = define_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

test_loader =  define_loaders(
        train_dataset=test_dataset,
        val_dataset=None,
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

model = define_model(name=config["MODEL"]["model_name"],
                     encoder_name=config["MODEL"]["encoder_name"],
                     in_channel=len(BANDS),
                     out_channels=len(BANDS),
                     activation=None)

nb_parameters = count_parameters(model=model)
logger.info("Number of parameters: {}".format(nb_parameters))
print("Number of parameters: {}".format(nb_parameters))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# # Define loss function
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=float(LEARNING_RATE))

# Training parameters
best_epoch = 0
best_val_loss = float('inf')
step = 0

# Initialize metrics dictionary with new metrics
dict_metrics = {
    "train_loss": [],
    "train_psnr": {band: [] for band in BANDS},
    "train_rmse": {band: [] for band in BANDS},
    "train_ssim": {band: [] for band in BANDS},
    "train_sam": {band: [] for band in BANDS},
    "val_loss": [],
    "val_psnr": {band: [] for band in BANDS},
    "val_rmse": {band: [] for band in BANDS},
    "val_ssim": {band: [] for band in BANDS},
    "val_sam": {band: [] for band in BANDS},
}

save_path = "checkpoints"
os.makedirs(save_path, exist_ok=True)
# Training loop
logger.info(f"Starting training for {NUM_EPOCHS} epochs")
weights_path = f"{save_path}/best_model.pth"

####################
# Training phase   #
####################


from training.metrics import MultiSpectralMetrics
import numpy as np
import matplotlib.pyplot as plt

# Initialize metrics trackers
train_metrics_tracker = MultiSpectralMetrics(bands=BANDS, device=device)
val_metrics_tracker = MultiSpectralMetrics(bands=BANDS, device=device)

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0

    # Reset metrics at the start of each epoch
    train_metrics_tracker.reset()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % BATCH_SIZE), ncols=100, colour='#3eedc4') as t:
        t.set_description(f'epoch: {epoch}/{NUM_EPOCHS - 1}')

        for batch_idx, (x_data, y_data) in enumerate(train_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Valid mask for loss calculation
            valid_mask = (y_data >= 0)

            # Forward pass
            outputs = model(x_data)

            # Update metrics for all bands
            train_metrics_tracker.update(outputs, y_data)

            # Calculate loss
            loss = criterion(outputs[valid_mask], y_data[valid_mask])

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            batch_loss = loss.item()
            train_loss += batch_loss

            # Update progress bar
            t.set_postfix(loss=batch_loss)
            t.update(x_data.size(0))

    # Compute average loss and metrics for the epoch
    avg_train_loss = train_loss / len(train_loader)
    dict_metrics['train_loss'].append(avg_train_loss)

    # Get all metrics
    train_epoch_metrics = train_metrics_tracker.compute()

    # Store metrics in the existing dictionary
    for band in BANDS:
        dict_metrics['train_psnr'][band].append(train_epoch_metrics[band]['psnr'])
        dict_metrics['train_rmse'][band].append(train_epoch_metrics[band]['rmse'])
        dict_metrics['train_ssim'][band].append(train_epoch_metrics[band]['ssim'])
        dict_metrics['train_sam'][band].append(train_epoch_metrics[band]['sam'])

    # Print a summary of training metrics
    logger.info(f"Epoch {epoch} training metrics:")
    for band in BANDS:
        logger.info(f"  {band}: PSNR={train_epoch_metrics[band]['psnr']:.2f}, "
              f"RMSE={train_epoch_metrics[band]['rmse']:.4f}, "
              f"SSIM={train_epoch_metrics[band]['ssim']:.4f}, "
              f"SAM={train_epoch_metrics[band]['sam']:.2f}°")

    ####################
    # Validation phase #
    ####################

    model.eval()
    val_loss = 0.0

    # Reset validation metrics
    val_metrics_tracker.reset()

    with torch.no_grad():
        with tqdm(total=len(val_dataset), ncols=100, colour='#f4d160') as t:
            t.set_description('validation')

            for batch_idx, (x_data, y_data) in enumerate(val_loader):
                x_data = x_data.to(device)
                y_data = y_data.to(device)
                valid_mask = (y_data >= 0)

                # Forward pass
                outputs = model(x_data)

                # Update validation metrics
                val_metrics_tracker.update(outputs, y_data)

                # Calculate loss
                loss = criterion(outputs[valid_mask], y_data[valid_mask])

                # Update statistics
                batch_loss = loss.item()
                val_loss += batch_loss

                # Update progress bar
                t.set_postfix(loss=batch_loss)
                t.update(x_data.size(0))

    # Compute average validation loss
    avg_val_loss = val_loss / len(val_loader)
    dict_metrics['val_loss'].append(avg_val_loss)

    # Get validation metrics
    val_epoch_metrics = val_metrics_tracker.compute()

    # Store validation metrics
    for band in BANDS:
        dict_metrics['val_psnr'][band].append(val_epoch_metrics[band]['psnr'])
        dict_metrics['val_rmse'][band].append(val_epoch_metrics[band]['rmse'])
        dict_metrics['val_ssim'][band].append(val_epoch_metrics[band]['ssim'])
        dict_metrics['val_sam'][band].append(val_epoch_metrics[band]['sam'])

    # Print a summary of validation metrics
    logger.info(f"Epoch {epoch} validation metrics:")
    for band in BANDS:
        logger.info(f"  {band}: PSNR={val_epoch_metrics[band]['psnr']:.2f}, "
              f"RMSE={val_epoch_metrics[band]['rmse']:.4f}, "
              f"SSIM={val_epoch_metrics[band]['ssim']:.4f}, "
              f"SAM={val_epoch_metrics[band]['sam']:.2f}°")

    # Log epoch results with all metrics
    train_metrics_str = ", ".join([f"Train PSNR {band}: {train_epoch_metrics[band]['psnr']:.4f}" for band in BANDS])
    val_metrics_str = ", ".join([f"Val PSNR {band}: {val_epoch_metrics[band]['psnr']:.4f}" for band in BANDS])


    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), f"{save_path}/best_model.pth")
        logger.info(f"Saved best model with Val Loss: {best_val_loss:.6f} at epoch {best_epoch+1}")


# Initialize test metrics dictionary
test_metrics = {
    "test_loss": 0.0,
    "test_psnr": {band: 0.0 for band in BANDS},
    "test_rmse": {band: 0.0 for band in BANDS},
    "test_ssim": {band: 0.0 for band in BANDS},
    "test_sam": {band: 0.0 for band in BANDS},
}

# Save final model
torch.save(model.state_dict(), f"{save_path}/final_model.pth")
logger.info(f"Training completed. Best Val Loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
logger.info("Model performance running on test data ...")
torch.cuda.empty_cache()

# Model Test - load best model weights
model = load_model_weights(model=model, filename=weights_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.eval()
test_loss = 0.0
test_metrics_tracker = MultiSpectralMetrics(bands=BANDS, device=device)

with torch.no_grad():
    with tqdm(total=(len(test_dataset) - len(test_dataset) % BATCH_SIZE), ncols=100, colour='#cc99ff') as t:
        t.set_description('testing')

        for batch_idx, (x_data, y_data) in enumerate(test_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            valid_mask = (y_data >= 0)

            # Forward pass
            outputs = model(x_data)

            # Update test metrics
            test_metrics_tracker.update(outputs, y_data)

            # Calculate loss
            loss = criterion(outputs[valid_mask], y_data[valid_mask])

            # Update statistics
            batch_loss = loss.item()
            test_loss += batch_loss

            # Update progress bar
            t.set_postfix(loss=batch_loss)
            t.update(x_data.size(0))

# Calculate average test loss
avg_test_loss = test_loss / len(test_loader)
test_metrics['test_loss'] = avg_test_loss

# Get test metrics
test_epoch_metrics = test_metrics_tracker.compute()

# Store test metrics
for band in BANDS:
    test_metrics['test_psnr'][band] = test_epoch_metrics[band]['psnr']
    test_metrics['test_rmse'][band] = test_epoch_metrics[band]['rmse']
    test_metrics['test_ssim'][band] = test_epoch_metrics[band]['ssim']
    test_metrics['test_sam'][band] = test_epoch_metrics[band]['sam']

    # Print metrics for each band
    print(f"Band {band}: Test PSNR: {test_epoch_metrics[band]['psnr']:.4f}, "
          f"RMSE: {test_epoch_metrics[band]['rmse']:.4f}, "
          f"SSIM: {test_epoch_metrics[band]['ssim']:.4f}, "
          f"SAM: {test_epoch_metrics[band]['sam']:.2f}°")

# Log test metrics
test_metrics_str = ", ".join([f"Test PSNR {band}: {test_metrics['test_psnr'][band]:.4f}" for band in BANDS])
logger.info(f"Test Loss: {avg_test_loss:.6f}, {test_metrics_str}")

# Save metrics
# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(dict_metrics['train_loss'], label='Train Loss')
plt.plot(dict_metrics['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(f"{save_path}/loss_curves.png")
logger.info(f"Loss curves saved to {save_path}/loss_curves.png")

# Plot metrics curves for each band
for band in BANDS:
    # PSNR curves
    plt.figure(figsize=(10, 6))
    plt.plot(dict_metrics['train_psnr'][band], label='Train PSNR')
    plt.plot(dict_metrics['val_psnr'][band], label='Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.title(f'Training and Validation PSNR for Band {band}')
    plt.legend()
    plt.savefig(f"{save_path}/psnr_curves_band_{band}.png")

    # RMSE curves
    plt.figure(figsize=(10, 6))
    plt.plot(dict_metrics['train_rmse'][band], label='Train RMSE')
    plt.plot(dict_metrics['val_rmse'][band], label='Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title(f'Training and Validation RMSE for Band {band}')
    plt.legend()
    plt.savefig(f"{save_path}/rmse_curves_band_{band}.png")

    # SSIM curves
    plt.figure(figsize=(10, 6))
    plt.plot(dict_metrics['train_ssim'][band], label='Train SSIM')
    plt.plot(dict_metrics['val_ssim'][band], label='Validation SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.title(f'Training and Validation SSIM for Band {band}')
    plt.legend()
    plt.savefig(f"{save_path}/ssim_curves_band_{band}.png")

    # SAM curves
    plt.figure(figsize=(10, 6))
    plt.plot(dict_metrics['train_sam'][band], label='Train SAM')
    plt.plot(dict_metrics['val_sam'][band], label='Validation SAM')
    plt.xlabel('Epochs')
    plt.ylabel('SAM (degrees)')
    plt.title(f'Training and Validation SAM for Band {band}')
    plt.legend()
    plt.savefig(f"{save_path}/sam_curves_band_{band}.png")

    logger.info(f"Metric curves for band {band} saved to {save_path}/")

# Optionally, test the model on a few validation samples
model.eval()
with torch.no_grad():
    for i, (x_data, y_data) in enumerate(test_loader):
        if i >= 5:  # Only process the first 5 samples
            break

        x_data = x_data.to(device)
        y_data = y_data.to(device)

        output = model(x_data)

        # Convert to numpy for visualization
        x_np = x_data.cpu().numpy()[0].transpose(1, 2, 0)  # First image in batch, CHW to HWC
        y_np = y_data.cpu().numpy()[0].transpose(1, 2, 0)
        pred_np = output.cpu().numpy()[0].transpose(1, 2, 0)

        # Clip values to valid range for visualization
        x_np = np.clip(x_np, 0, 1)
        y_np = np.clip(y_np, 0, 1)
        pred_np = np.clip(pred_np, 0, 1)

        # Plot and save
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(x_np)
        axs[0].set_title('L1C Input')
        axs[1].imshow(pred_np)
        axs[1].set_title('Model Output')
        axs[2].imshow(y_np)
        axs[2].set_title('L2A Ground Truth')

        plt.savefig(f"{save_path}/sample_{i}_prediction.png")
        plt.close()

logger.info("Testing completed. Sample predictions saved.")

# Save all metrics to CSV for easy analysis
import pandas as pd

# Training and validation metrics per epoch
for metric_type in ['psnr', 'rmse', 'ssim', 'sam']:
    df_data = {
        'epoch': list(range(NUM_EPOCHS))
    }

    for phase in ['train', 'val']:
        for band in BANDS:
            df_data[f'{phase}_{band}'] = dict_metrics[f'{phase}_{metric_type}'][band]

    df = pd.DataFrame(df_data)
    df.to_csv(f"{save_path}/{metric_type}_metrics.csv", index=False)
    logger.info(f"Saved {metric_type} metrics to {save_path}/{metric_type}_metrics.csv")

# Test metrics summary
test_summary = {
    'band': BANDS,
    'psnr': [test_metrics['test_psnr'][band] for band in BANDS],
    'rmse': [test_metrics['test_rmse'][band] for band in BANDS],
    'ssim': [test_metrics['test_ssim'][band] for band in BANDS],
    'sam': [test_metrics['test_sam'][band] for band in BANDS]
}

df_test = pd.DataFrame(test_summary)
df_test.to_csv(f"{save_path}/test_metrics_summary.csv", index=False)
logger.info(f"Saved test metrics summary to {save_path}/test_metrics_summary.csv")