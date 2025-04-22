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
from torchmetrics.image import PeakSignalNoiseRatio

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

dict_metrics = {
    "train_loss": [],
    "train_psnr": {band: [] for band in BANDS},
    "val_loss": [],
    "val_psnr": {band: [] for band in BANDS},
}

save_path = "checkpoints"
os.makedirs(save_path, exist_ok=True)
# Training loop
logger.info(f"Starting training for {NUM_EPOCHS} epochs")
weights_path = f"{save_path}/best_model.pth"

####################
# Training phase   #
####################

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    psnr_channels = [
        PeakSignalNoiseRatio(data_range=1.0).to(device) for _ in range(len(BANDS))
    ]

    with tqdm(total=(len(train_dataset) - len(train_dataset) % BATCH_SIZE), ncols=100, colour='#3eedc4') as t:
        t.set_description('epoch: {}/{}'.format(epoch, NUM_EPOCHS - 1))

        for batch_idx, (x_data, y_data) in enumerate(train_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)

            # Clear gradients
            optimizer.zero_grad()

            valid_mask = (y_data >= 0)
            # Forward pass
            outputs = model(x_data)

            ## Compute metrics for each bands
            for c, band in enumerate(BANDS):
                # Extract channel c for both outputs and targets.
                # E.g -> [32, 1, 1024, 1024]
                outputs_c = outputs[:, c, :, :]
                y_c = y_data[:, c, :, :]

                # Create channel-wise valid mask. This mask is True for valid pixels.
                valid_mask_c = (y_c >= 0)

                # Select only the valid pixels.
                outputs_valid_c = outputs_c[valid_mask_c]
                y_valid_c = y_c[valid_mask_c]

                # Update the metrics for to the channel.
                psnr_channels[c].update(outputs_valid_c, y_valid_c)

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

    avg_train_loss = train_loss / len(train_loader)
    dict_metrics['train_loss'].append(avg_train_loss)

    epoch_psnr_train = {}
    for c, band in enumerate(BANDS):
        avg_psnr = psnr_channels[c].compute().item()
        epoch_psnr_train[band] = avg_psnr
    dict_metrics['train_psnr'] = {band: dict_metrics['train_psnr'][band] + [epoch_psnr_train[band]] for band in BANDS}

    ####################
    # Validation phase #
    ####################

    model.eval()
    val_loss = 0.0
    psnr_channels = [
        PeakSignalNoiseRatio(data_range=1.0).to(device) for _ in range(len(BANDS))
    ]
    criterion = nn.MSELoss()
    with torch.no_grad():
        with tqdm(total=len(val_dataset), ncols=100, colour='#f4d160') as t:
            t.set_description('validation')

            for batch_idx, (x_data, y_data) in enumerate(val_loader):
                x_data = x_data.to(device)
                y_data = y_data.to(device)
                valid_mask = (y_data >= 0)
                # Forward pass
                outputs = model(x_data)
                ## Compute metrics for each bands
                for c, band in enumerate(BANDS):
                    # Extract channel c for both outputs and targets.
                    outputs_c = outputs[:, c, :, :]
                    y_c = y_data[:, c, :, :]
                    # Create channel-wise valid mask. This mask is True for valid pixels.
                    valid_mask_c = (y_c >= 0)
                    # Select only the valid pixels.
                    outputs_valid_c = outputs_c[valid_mask_c]
                    y_valid_c = y_c[valid_mask_c]
                    # Update the metrics for this channel.
                    psnr_channels[c].update(outputs_valid_c, y_valid_c)

                loss = criterion(outputs[valid_mask], y_data[valid_mask])

                # Update statistics
                batch_loss = loss.item()
                val_loss += batch_loss

                # Update progress bar
                t.set_postfix(loss=batch_loss)
                t.update(x_data.size(0))

    avg_val_loss = val_loss / len(val_loader)
    dict_metrics['val_loss'].append(avg_val_loss)

    epoch_psnr_val = {}
    for c, band in enumerate(BANDS):
        avg_psnr = psnr_channels[c].compute().item()
        epoch_psnr_val[band] = avg_psnr

    dict_metrics['val_psnr'] = {band: dict_metrics['val_psnr'][band] + [epoch_psnr_val[band]] for band in BANDS}

    # Log epoch results
    psnr_str_train = ", ".join([f"Train PSNR {band}: {epoch_psnr_train[band]:.4f}" for band in BANDS])
    psnr_str_val = ", ".join([f"Val PSNR {band}: {epoch_psnr_val[band]:.4f}" for band in BANDS])
    logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, {psnr_str_train}, {psnr_str_val}")
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), f"{save_path}/best_model.pth")
        logger.info(f"Saved best model with Val Loss: {best_val_loss:.6f} at epoch {best_epoch+1}")


test_metrics = {
    "test_loss": [],
    "test_psnr": {band: [] for band in BANDS},
}
# Save final model
torch.save(model.state_dict(), f"{save_path}/final_model.pth")
logger.info(f"Training completed. Best Val Loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
logger.info("Model performance running on test data ...")
torch.cuda.empty_cache()
# Model Test
model = load_model_weights(model=model, filename=weights_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.eval()
test_loss = 0.0  # Changed variable name from val_loss to test_loss
psnr_channels = [
    PeakSignalNoiseRatio(data_range=1.0).to(device) for _ in range(len(BANDS))
]
criterion = nn.MSELoss()
with torch.no_grad():
    with tqdm(total=len(test_dataset), ncols=100, colour='#cc99ff') as t:
        t.set_description('testing')  # Changed from 'validation' to 'testing'

        for batch_idx, (x_data, y_data) in enumerate(test_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            valid_mask = (y_data >= 0)

            # Forward pass
            outputs = model(x_data)

            ## Compute metrics for each bands
            for c, band in enumerate(BANDS):
                # Extract channel c for both outputs and targets.
                outputs_c = outputs[:, c, :, :]
                y_c = y_data[:, c, :, :]

                # Create channel-wise valid mask. This mask is True for valid pixels.
                valid_mask_c = (y_c >= 0)

                # Select only the valid pixels.
                outputs_valid_c = outputs_c[valid_mask_c]
                y_valid_c = y_c[valid_mask_c]

                # Update the metrics for this channel.
                psnr_channels[c].update(outputs_valid_c, y_valid_c)

            loss = criterion(outputs[valid_mask], y_data[valid_mask])

            # Update statistics
            batch_loss = loss.item()
            test_loss += batch_loss

            # Update progress bar
            t.set_postfix(loss=batch_loss)
            t.update(x_data.size(0))

avg_test_loss = test_loss / len(test_loader)
test_metrics['test_loss'] = avg_test_loss  # You might need to update this line

test_psnr = {}
for c, band in enumerate(BANDS):
    avg_psnr = psnr_channels[c].compute().item()
    test_psnr[band] = avg_psnr
    print(f"Band {band}: Test PSNR: {avg_psnr:.4f}")

test_metrics["test_psnr"] = test_psnr

psnr_str_test = ", ".join([f"Test PSNR {band}: {test_psnr[band]:.4f}" for band in BANDS])
logger.info(f"Test Loss: {avg_test_loss:.6f}, {psnr_str_test}")

# Save metrics
import matplotlib.pyplot as plt
import numpy as np

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

# Plot PSNR curves for each band
for band in BANDS:
    plt.figure(figsize=(10, 6))
    plt.plot(dict_metrics['train_psnr'][band], label='Train PSNR')
    plt.plot(dict_metrics['val_psnr'][band], label='Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.title(f'Training and Validation PSNR for Band {band}')
    plt.legend()
    plt.savefig(f"{save_path}/psnr_curves_band_{band}.png")
    logger.info(f"PSNR curves for band {band} saved to {save_path}/psnr_curves_band_{band}.png")

# Optionally, test the model on a few validation samples
model.eval()
with torch.no_grad():
    for i, (x_data, y_data) in enumerate(test_loader):

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